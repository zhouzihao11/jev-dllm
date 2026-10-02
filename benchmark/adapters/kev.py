"""Input-only native Kev inference; runtime imports belong in the Kev child."""

import math
from pathlib import Path
import sys
import time

from common import load, option_count, require


class Adapter:
    def __init__(self, args):
        require(args.batch_size == 1, "Kev native quality evaluation requires batch-size 1")
        require(args.dtype == "float32", "Kev native quality evaluation requires float32")
        source = Path(args.kev_source).expanduser().resolve()
        base = Path(args.base_path).expanduser().resolve()
        checkpoint_path = Path(args.model_path).expanduser().resolve()
        require((source / "kev" / "checkpoint.py").is_file(), "--kev-source must contain the kev package")
        require(base.is_dir() and checkpoint_path.is_dir(), "Kev requires existing local base and adapter directories")
        sys.path.insert(0, str(source))
        import torch
        from kev import api, checkpoint, model
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import ContextOverflow

        origins = {module.__name__: str(Path(module.__file__).resolve())
                   for module in (api, checkpoint, model)}
        for name, origin in origins.items():
            require(Path(origin) == source / "kev" / (name.rsplit(".", 1)[-1] + ".py"),
                    "Native Kev module not loaded from configured source: " + name)
        self.torch = torch
        self.request, self.to_record = SystemOneRequest, to_record
        self.context_overflow, self.user_tokens = ContextOverflow, model.user_tokens
        self.device, self.max_length = args.device, args.max_length
        self.cuda = str(args.device).startswith("cuda")
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        ck = Checkpoint(str(checkpoint_path))
        original_base, original_revision = ck.meta.base, ck.meta.base_revision
        config = load(base / "config.json")
        base_metadata = {key: config[key] for key in
                         ("_name_or_path", "name_or_path", "_commit_hash", "revision", "base_revision")
                         if config.get(key)}
        for key in ("_name_or_path", "name_or_path"):
            if key in base_metadata:
                require(base_metadata[key] in (original_base, str(base)),
                        "Local base config name differs from checkpoint base")
        for key in ("_commit_hash", "revision", "base_revision"):
            if key in base_metadata:
                require(base_metadata[key] == original_revision,
                        "Local base config revision differs from checkpoint base_revision")
        require(not ck.full, "Kev adapter backend expects LoRA weights and head.pt")
        require(ck.meta.weights_dtype == "fp32", "Checkpoint would override the FP32 quality protocol")
        temperature = ck.meta.temperature if args.kev_temperature is None else args.kev_temperature
        require(math.isfinite(temperature) and temperature > 0, "Kev temperature must be finite and positive")
        # Only redirect the in-memory loader; never rewrite head.pt or base config.
        ck.meta.base, ck.meta.base_revision = str(base), None
        opts = LoadOptions(dtype=torch.float32, merge=True, attn="sdpa", backend="torch",
                           cuda_graphs=False, fused=False, lora_scale=1.0,
                           temperature=args.kev_temperature)
        self.tok, self.model = ck.load(args.device, opts)
        require(self.model.dtype == "float32", "Native Kev did not load in FP32")
        self.temperature = float(self.model.head.temperature)
        self.runtime = dict(
            backend="kev", protocol="native_kev_fp32_single_question", backend_engine=self.model.backend,
            requested_dtype=args.dtype, dtype=self.model.dtype, device=str(args.device),
            backend_python=sys.executable, native_code_origin=origins,
            base_path=str(base), model_path=str(checkpoint_path),
            original_base=original_base, base_revision=original_revision,
            local_base_config_metadata=base_metadata,
            base_revision_check="matched_config" if any(k in base_metadata for k in
                                ("_commit_hash", "revision", "base_revision")) else "not_recorded_in_config",
            head_loaded=True, head_dim=ck.meta.head_dim, model_dim=self.model.lm.config.hidden_size,
            checkpoint_weights_dtype=ck.meta.weights_dtype, lora=ck.meta.lora,
            option_isolation=self.model.option_isolation, hybrid=self.model.hybrid,
            checkpoint_temperature=float(ck.meta.temperature), temperature=self.temperature,
            calibration="checkpoint" if args.kev_temperature is None else "explicit_temperature_diagnostic",
            benchmark_temperature_fit=False,
            load_options=dict(dtype="float32", merge=True, attn="sdpa", backend="torch",
                              cuda_graphs=False, fused=False, lora_scale=1.0,
                              temperature=args.kev_temperature),
            context=dict(max_state=args.max_length, max_branch=args.max_length,
                         max_packed=args.max_length, strict=True),
            timing_scope="synchronized native forward (including head), softmax and CPU tensor copies; "
                         "excludes encoding, JSON IO and model loading; one supported decision per batch",
            logits_location="input_metadata.logits (native head logits at inference_temperature)",
        )

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
        elif str(self.device) == "mps":
            self.torch.mps.synchronize()

    def predict_batch(self, inputs, batch_id):
        require(len(inputs) == 1, "Kev supports exactly one decision per batch")
        item = inputs[0]
        require(set(item) == {"state", "qdef"}, "Adapter accepts state/qdef ONLY")
        count = option_count(item["qdef"])
        rec, meta = self.to_record(self.request(state=item["state"], questions={"decision": item["qdef"]}))
        keys = meta[0]["keys"]
        require(len(keys) == count, "Native option count differs from input")
        try:
            enc = self.model.encode(self.tok, rec, max_state=self.max_length,
                                    max_branch=self.max_length, strict=True)
            length = len(enc["ids"])
            if length > self.max_length:
                raise self.context_overflow(f"packed request exceeds {self.max_length} tokens: {length}")
        except self.context_overflow as exc:
            # The strict encoder raises before returning ids. Count the same native
            # tokens and delimiters without truncating or allocating a model mask.
            q = rec["questions"][0]
            size = lambda text: len(self.user_tokens(self.tok, text))
            length = 1 + size(rec["state"]) + 1 + size(q["instr"]) + 1
            length += sum(2 + size(option) for option in q["options"])
            return [dict(status="unsupported_length", length=length, probs=None, pred_idx=None,
                         error=str(exc), batch_id=batch_id, actual_batch_size=0,
                         forward_seconds=0.0)], None

        with self.torch.no_grad():
            self.sync()
            started = time.perf_counter()
            logits = self.model.forward(enc)
            ps = [self.torch.softmax(z, -1).cpu() for z in logits]
            zs = [z.float().cpu() for z in logits]
            self.sync()
            seconds = time.perf_counter() - started
        require(len(ps) == len(zs) == 1 and ps[0].numel() == count, "Native output shape mismatch")
        probs, calibrated_logits = ps[0].tolist(), zs[0].tolist()
        require(all(math.isfinite(v) for v in probs + calibrated_logits), "Nonfinite native Kev output")
        prediction = dict(status="ok", length=length, probs=probs,
                          pred_idx=max(range(len(probs)), key=probs.__getitem__),
                          batch_id=batch_id, actual_batch_size=1, forward_seconds=seconds,
                          input_metadata=dict(option_keys=keys, logits=calibrated_logits,
                                              inference_temperature=self.temperature,
                                              logits_kind="native_head_calibrated"))
        return [prediction], dict(batch_id=batch_id, batch_size=1, seconds=seconds)
