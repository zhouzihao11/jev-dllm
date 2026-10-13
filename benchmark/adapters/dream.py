"""Input-only, one-forward Dream scoring with the historical generic noul prompt."""

import time

from common import inherited, require


class Adapter:
    def __init__(self, args):
        require(args.batch_size == 1, "Dream requires --batch-size 1")
        from transformers import AutoModel, AutoTokenizer

        self.module = inherited()
        torch = self.module.torch
        self.device = torch.device(args.device)
        self.tok = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
        self.model.to(self.device).eval()
        config_mask = getattr(self.model.config, "mask_token_id", None)
        tokenizer_mask = self.tok.mask_token_id
        require(config_mask is not None or tokenizer_mask is not None,
                "Dream config/tokenizer must declare a mask token; no fallback")
        mask_id = config_mask if config_mask is not None else tokenizer_mask
        require(type(mask_id) is int and 0 <= mask_id < self.model.config.vocab_size,
                "Invalid Dream mask token ID")
        self.mapping, self.mask_id, self.mask_text = self.module.token_mapping(self.tok, self.model)
        limits = [args.max_length]
        for limit in (getattr(self.model.config, "max_position_embeddings", None),
                      getattr(self.tok, "model_max_length", None)):
            if type(limit) is int and 0 < limit < 10**9:
                limits.append(limit)
        self.limit = min(limits)
        self.runtime = dict(
            loader="transformers.AutoModel.from_pretrained", model_class=type(self.model).__name__,
            tokenizer_class=type(self.tok).__name__, dtype=str(self.model.dtype),
            model_dtype=str(self.model.dtype), device=str(self.device),
            mask_id=self.mask_id, mask_token=self.mask_text, answer_token_ids=self.mapping,
            logit_columns=["Yes", "No"], temperature=1.0,
            attention_implementation=getattr(self.model.config, "_attn_implementation", None),
            attention_module_classes=sorted({type(module).__name__ for module in self.model.modules()
                                             if "attention" in type(module).__name__.lower()}),
            attention_mask="full", attention="full bidirectional SDPA; is_causal=False",
            position_convention="raw unshifted logits at mask_position - 1", logit_position_offset=-1,
            renderer="oldgeneric_noul_v1", noul_criteria_rendering="ignored in generic noul prompt",
            system="You are a helpful assistant.", add_generation_prompt=False, enable_thinking=False,
            generation=False, denoising=False, use_cache=False, full_vocab_forward=True,
            candidate_projection=False, forwards_per_supported_decision=1, warmup_forwards=0,
            batch_size=1, padding=False, truncation=False, quantization=False,
            requested_max_length=args.max_length, context_limit=self.limit,
            timing="synchronized native full-vocabulary forward and Yes/No gather; excludes tokenization, "
                   "H2D, D2H, CPU softmax and IO")

    def prepare(self, item, batch_id):
        require(set(item) == {"state", "qdef"}, "Adapter accepts state/qdef ONLY")
        q = self.module.to_internal(item["qdef"])
        options = self.module.render_options(q)
        require(q["t"] in ("choice", "score", "noul") and options, "Unsupported decision")
        state = self.module.serialize_state(item["state"])
        record = dict(status="ok", length=None, probs=None, pred_idx=None, raw_logits=None,
                      forward_seconds=0.0, batch_id=batch_id, actual_batch_size=0,
                      input_options=options)
        if any(self.mask_text in text for text in [state, q["ins"]] + options):
            record.update(status="unsupported_source_mask", error="Source contains literal mask token")
            return record, None
        user = "State:\n%s\n\nQuestion:\n%s\n\n" % (state, q["ins"])
        if q["t"] == "noul":
            user += "Answer Yes or No.\nAnswer:" + self.mask_text
            count = 1
        else:
            user += "For each option, mark Yes if it answers the question, otherwise No.\n"
            user += "\n".join(option + ": " + self.mask_text for option in options)
            count = len(options)
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        parts = prompt.split(self.mask_text)
        require(len(parts) == count + 1, "Chat template changed the number of masks")
        ids, positions = [], []
        for i, part in enumerate(parts):
            ids.extend(self.tok.encode(part, add_special_tokens=False, truncation=False))
            if i < count:
                positions.append(len(ids))
                ids.append(self.mask_id)
        require(ids.count(self.mask_id) == count, "Unexpected source/template mask IDs")
        require(all(position > 0 for position in positions), "Dream mask positions must be positive")
        record.update(length=len(ids), input_metadata=dict(
            mask_positions=positions, prediction_positions=[position - 1 for position in positions],
            renderer="oldgeneric_noul_v1"))
        if len(ids) > self.limit:
            record.update(status="unsupported_length",
                          error="Input exceeds context limit %d; no truncation" % self.limit)
            return record, None
        return record, (ids, positions)

    def predict_batch(self, inputs, batch_id):
        require(len(inputs) == 1, "Dream accepts exactly one decision per batch")
        record, prepared = self.prepare(inputs[0], batch_id)
        if prepared is None:
            return [record], None
        ids, positions = prepared
        torch, np = self.module.torch, self.module.np
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        prediction_positions = torch.tensor([p - 1 for p in positions], device=self.device)
        candidates = torch.tensor([self.mapping["Yes"], self.mapping["No"]], device=self.device)
        with torch.inference_mode():
            self.module.synchronize(self.device)
            started = time.perf_counter()
            output = self.model(input_ids=input_ids, attention_mask="full", use_cache=False, return_dict=True)
            # Native forward is unshifted; Dream generation's right shift is applied here only once.
            z_tensor = output.logits[0, prediction_positions[:, None], candidates[None, :]].float()
            self.module.synchronize(self.device)
            elapsed = time.perf_counter() - started
            z = z_tensor.cpu().numpy().astype(float)
            del output, z_tensor
        require(z.shape == (len(positions), 2) and np.isfinite(z).all(), "Invalid Yes/No logits")
        logodds = z[:, 0] - z[:, 1]
        require(np.isfinite(logodds).all(), "Nonfinite logodds")
        pairs = np.asarray([self.module.softmax_t(pair, 1.0) for pair in z])
        for pair in pairs:
            self.module.validate_probs(pair)
        p = pairs[0, ::-1] if inputs[0]["qdef"]["type"] == "noul" else self.module.softmax_t(logodds, 1.0)
        self.module.validate_probs(p)
        require(len(p) == len(record["input_options"]), "Probability/options mismatch")
        record.update(raw_logits=z.tolist(), yes_no_probs=pairs.tolist(), logodds=logodds.tolist(),
                      probs=p.tolist(), pred_idx=int(np.argmax(p)), forward_seconds=elapsed, actual_batch_size=1)
        timing = dict(batch_id=batch_id, batch_size=1, forwards=1, forward_seconds=elapsed,
                      nonpad_tokens=len(ids), padded_tokens=len(ids))
        return [record], timing
