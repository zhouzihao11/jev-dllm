"""Native Dream decision-level supervised FSDP training; see docs/dream_s1.md."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any
import uuid

from _benchmark_bootstrap import bootstrap

bootstrap()

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, ShardingStrategy, StateDictType,
    ShardedStateDictConfig, ShardedOptimStateDictConfig, FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModel, AutoTokenizer

from dream_training import CandidateWrapper, memory
from shared_yesno_supervised import (
    load_structured_items, prediction_record, summarize_predictions, supervised_terms,
)


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temp, value)
    os.replace(temp, path)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def rank_zero(action) -> Any:
    payload: list[Any] = [None, None]
    if dist.get_rank() == 0:
        try:
            payload[0] = action()
        except Exception as exc:
            payload[1] = repr(exc)
    dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise RuntimeError(payload[1])
    return payload[0]


def data_inputs(tok, base, max_length):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from adapters.dream import Adapter
    from common import inherited
    module = inherited()
    tokens = module.token_mapping(tok, base)
    limits = [max_length]
    for value in (getattr(base.config, "max_position_embeddings", None), tok.model_max_length):
        if type(value) is int and 0 < value < 10**9:
            limits.append(value)

    def renderer(tokenizer, state, q, mask_id, mask_text, limit):
        context = SimpleNamespace(module=module, tok=tokenizer, mask_id=mask_id,
                                  mask_text=mask_text, limit=limit)
        record, prepared = Adapter.prepare(context, {"state": state, "qdef": q}, 0)
        if prepared is None:
            raise ValueError("%s: %s" % (record["status"], record.get("error")))
        ids, positions = prepared
        return ids, positions, record["input_options"], tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return tokens, min(limits), renderer


def forward(model, item, device):
    return model(torch.tensor([item["ids"]], device=device),
                 torch.tensor(item["candidates"], device=device), item["positions"], item["noul"])


@contextmanager
def dream_generation_export_compat(config):
    cls = type(config)
    if cls.__name__ != "DreamGenerationConfig" or "strict" in inspect.signature(cls.validate).parameters:
        yield
        return
    original = cls.validate

    def validate(self, is_init=False, *, strict=False):
        return original(self, is_init=is_init)

    # Patch the class, not serializable instance attributes, only during HF export.
    cls.validate = validate
    try:
        yield
    finally:
        cls.validate = original


def preserve_local_code(source, target):
    # Copy only root Python files, after HF dynamic-module saving, never source weights.
    source = Path(source).resolve()
    copied = set()
    for path in source.glob("*.py"):
        if path.resolve().parent != source or not path.is_file():
            raise ValueError("Custom code must be a file within the pinned model root: " + str(path))
        shutil.copyfile(path, target / path.name)
        copied.add(path.name)
    for filename in ("config.json", "tokenizer_config.json"):
        original = read_json(source / filename) if (source / filename).exists() else {}
        saved = read_json(target / filename)
        mapping = original.get("auto_map", saved.get("auto_map", {}))

        def local_reference(value):
            if value is None:
                return None
            if isinstance(value, (tuple, list)):
                return [local_reference(x) for x in value]
            reference = value.split("--")[-1]
            module, _ = reference.rsplit(".", 1)
            if not module.isidentifier() or module + ".py" not in copied:
                raise ValueError("Missing root custom module for auto_map: " + value)
            return reference
        if mapping:
            saved["auto_map"] = {key: local_reference(value) for key, value in mapping.items()}
            atomic_json(target / filename, saved)
    generation = source / "generation_config.json"
    if generation.is_file() and not (target / generation.name).exists():
        shutil.copyfile(generation, target / generation.name)
    if any(not (target / name).is_file() for name in copied):
        raise FileNotFoundError("Incomplete custom-code export")


def run(a, device, emit):
    rank, world = dist.get_rank(), dist.get_world_size()
    if a.global_batch % world:
        raise ValueError("global-batch must be divisible by world size (microbatch is one)")
    output = Path(a.output_dir)

    def prepare():
        if a.resume:
            return read_json(output / "run.json")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Fresh runs require an empty output-dir")
        output.mkdir(parents=True, exist_ok=True)
        return dict(run_id=uuid.uuid4().hex)
    run_meta = rank_zero(prepare)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    emit("init", settings=vars(a), world_size=world)
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    base = AutoModel.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True,
                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    base.requires_grad_(True)
    base.config.use_cache = base.model.config.use_cache = False
    tokens, context_limit, renderer = data_inputs(tok, base, a.max_length)
    train = load_structured_items(a.train_data, tok, context_limit, tokens, a.train_limit, renderer)
    dev = load_structured_items(a.dev_data, tok, context_limit, tokens, a.dev_limit, renderer)
    if not train or not dev or any(x["split"] != "train" for x in train) or any(
            x["split"] not in ("dev", "validation") for x in dev):
        raise ValueError("Expected nonempty canonical train/dev splits")
    batches = math.ceil(len(train) / a.global_batch)
    total_steps = batches * a.epochs
    warmup = math.ceil(total_steps * a.warmup_ratio)
    config = {k: v for k, v in vars(a).items() if k not in (
        "resume", "output_dir", "max_updates", "save_every", "log_every")}
    config.update(world_size=world, train_count=len(train), dev_count=len(dev),
                  total_steps=total_steps, context_limit=context_limit, tokens=tokens,
                  renderer="oldgeneric_noul_v1", temperature=1.0)
    config = json.loads(json.dumps(config))
    if a.resume and run_meta["config"] != config:
        raise ValueError("Resume world/model/data/core training parameters differ")
    run_meta["config"] = config
    if not a.resume:
        rank_zero(lambda: write_json(output / "run.json", run_meta))
    emit("tokenized", train_count=len(train), dev_count=len(dev), schedule_updates=total_steps,
         context_limit=context_limit, micro_batch=1, accumulation=a.global_batch // world)
    base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.to(device)
    layer_classes = {type(layer) for layer in base.model.layers}
    if len(layer_classes) != 1 or next(iter(layer_classes)).__name__ != "DreamDecoderLayer":
        raise ValueError("Expected native DreamDecoderLayer")
    model = FSDP(CandidateWrapper(base), sharding_strategy=ShardingStrategy.FULL_SHARD,
                 auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes),
                 device_id=device, limit_all_gathers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)

    def lr_factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        floor = a.min_lr / a.learning_rate
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    emit("FSDP_ready", schedule_updates=total_steps, warmup_steps=warmup)
    checkpoint_context = lambda: FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True))
    step, epoch, cursor, last_eval, best = 0, 0, 0, -1, None

    def owned(name, marker):
        path = output / name
        if (Path(name).name != name or name.endswith(".tmp")
                or path.is_symlink() or path.resolve().parent != output):
            raise ValueError("Unsafe output generation: " + name)
        if read_json(path / marker)["run_id"] != run_meta["run_id"]:
            raise ValueError("Generation belongs to another run")
        return path

    if a.resume:
        resume = Path(a.resume)
        if resume.is_file():
            pointer = read_json(resume)
            resume = resume.parent / pointer["generation"]
        resume = owned(resume.name, "complete.json") if resume.parent == output else None
        if resume is None:
            raise ValueError("Resume generation must be inside output-dir")
        complete = read_json(resume / "complete.json")
        saved = torch.load(resume / ("rank%d.pt" % rank), map_location="cpu", weights_only=False)
        if (saved["config"] != config or saved["run_id"] != run_meta["run_id"]
                or complete["global_step"] != saved["global_step"] or complete["world_size"] != world):
            raise ValueError("Resume checkpoint configuration differs")
        with checkpoint_context():
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, saved["optimizer"]))
        scheduler.load_state_dict(saved["scheduler"])
        step, epoch, cursor = saved["global_step"], saved["epoch"], saved["next_batch"]
        best, last_eval = saved["best"], saved["last_eval"]
        if (not 0 <= epoch <= a.epochs or not 0 <= cursor < batches or (epoch == a.epochs and cursor)
                or step != epoch * batches + cursor or scheduler.last_epoch != step):
            raise ValueError("Invalid optimizer-boundary data cursor/scheduler")
        # A published best may be newer than the recovered previous checkpoint.
        if (output / "best.json").exists():
            published = read_json(output / "best.json")
            owned(published["path"], "training_metadata.json")
            if best is None or published["kl"] < best["kl"]:
                best = published
        if best is not None:
            owned(best["path"], "training_metadata.json")
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        del saved
        emit("resume", global_step=step, epoch=epoch, next_batch=cursor, best=best)

    def generation_name(prefix):
        name = "%s-step-%d" % (prefix, step) if prefix else "step-%d" % step
        if (output / name).exists() or (output / (name + ".tmp")).exists():
            name += "-recovery-" + uuid.uuid4().hex
        (output / (name + ".tmp")).mkdir()
        return name

    def export_best(kl, eval_epoch, predictions_file):
        nonlocal best
        emit("export_begin", global_step=step)
        name = rank_zero(lambda: generation_name("best"))
        temp = output / (name + ".tmp")
        selected = dict(path=name, step=step, kl=kl, epoch=eval_epoch, dev_count=len(dev),
                        criterion="mean_dev_kl", temperature=1.0, renderer="oldgeneric_noul_v1",
                        run_id=run_meta["run_id"], predictions_file=predictions_file)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()

        def publish():
            native = {key.removeprefix("base."): value for key, value in state.items()}
            with dream_generation_export_compat(base.generation_config):
                base.save_pretrained(temp, state_dict=native, safe_serialization=True, max_shard_size="5GB")
            tok.save_pretrained(temp)
            preserve_local_code(a.model_path, temp)
            write_json(temp / "training_metadata.json", selected)
            os.replace(temp, output / name)
            atomic_json(output / "best.json", selected)
        rank_zero(publish)
        del state
        best = selected
        dist.barrier()
        emit("export", global_step=step, best=best)

    def evaluate(eval_epoch, reason):
        nonlocal last_eval
        if last_eval == step:
            return
        emit("dev_begin", global_step=step, reason=reason)
        was_training = model.training
        model.eval()
        records = []
        with torch.no_grad():
            for index in range(rank, math.ceil(len(dev) / world) * world, world):
                item = dev[min(index, len(dev) - 1)]
                z = forward(model, item, device)
                if index < len(dev):
                    records.append((index, prediction_record(item, z)))
        shards: list[Any] = [None] * world
        dist.gather_object(records, shards if rank == 0 else None, dst=0)

        def publish():
            rows = [row for _, row in sorted(pair for shard in shards for pair in shard)]
            metrics = summarize_predictions(rows)
            if metrics["kl"] is None or not math.isfinite(metrics["kl"]):
                raise FloatingPointError("Nonfinite mean dev KL")
            stem = "dev-step-%d" % step
            if (output / (stem + ".json")).exists():
                stem += "-recovery-" + uuid.uuid4().hex
            write_json(output / (stem + ".json"), dict(step=step, epoch=eval_epoch, reason=reason, metrics=metrics))
            with open(output / (stem + "-predictions.jsonl"), "x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            return metrics, stem + "-predictions.jsonl"
        metrics, predictions_file = rank_zero(publish)
        if best is None or metrics["kl"] < best["kl"]:
            export_best(metrics["kl"], eval_epoch, predictions_file)
        last_eval = step
        model.train(was_training)
        emit("dev", global_step=step, epoch=eval_epoch, metrics=metrics, best=best)

    def save_checkpoint():
        emit("save_begin", global_step=step)
        name = rank_zero(lambda: generation_name(""))
        temp = output / (name + ".tmp")
        metadata = dict(run_id=run_meta["run_id"], global_step=step, world_size=world,
                        epoch=epoch, next_batch=cursor, best=best)
        with checkpoint_context():
            state = dict(metadata, config=config, model=model.state_dict(),
                         optimizer=FSDP.optim_state_dict(model, optimizer), scheduler=scheduler.state_dict(),
                         last_eval=last_eval, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                                     torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
            torch.save(state, temp / ("rank%d.pt" % rank))
            del state
        dist.barrier()

        def publish():
            prior = read_json(output / "latest.json") if (output / "latest.json").exists() else {}
            previous = prior.get("generation")
            if previous:
                owned(previous, "complete.json")
            write_json(temp / "complete.json", metadata)
            os.replace(temp, output / name)
            atomic_json(output / "latest.json", dict(metadata, generation=name, previous=previous))
            return prior
        prior = rank_zero(publish)
        dist.barrier()

        def prune():
            retired = prior.get("previous")
            if retired and retired not in (name, prior.get("generation")):
                shutil.rmtree(owned(retired, "complete.json"))
            keep_best = {best["path"]} if best else set()
            for generation in (name, prior.get("generation")):
                if generation:
                    value = read_json(owned(generation, "complete.json") / "complete.json")["best"]
                    if value:
                        keep_best.add(value["path"])
            for path in output.glob("best-step-*"):
                marker = path / "training_metadata.json"
                if path.name.endswith(".tmp") or path.name in keep_best or not marker.is_file():
                    continue
                if read_json(marker).get("run_id") == run_meta["run_id"]:
                    shutil.rmtree(owned(path.name, "training_metadata.json"))
        rank_zero(prune)
        dist.barrier()
        emit("save", global_step=step, generation=name, previous=prior.get("generation"))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if not a.resume:
        evaluate(0, "initial")
    cap = min(total_steps, a.max_updates) if a.max_updates else total_steps
    while step < cap:
        emit("epoch", epoch=epoch, next_batch=cursor)
        indices = list(range(len(train)))
        random.Random(a.seed + epoch).shuffle(indices)
        for bi in range(cursor, batches):
            started = time.perf_counter()
            window = indices[bi * a.global_batch:(bi + 1) * a.global_batch]
            real = len(window)
            padded = window + [None] * ((-real) % world)
            sums = torch.zeros(4, dtype=torch.float64, device=device)
            lr = optimizer.param_groups[0]["lr"]
            for index in padded[rank::world]:
                item = train[window[0] if index is None else index]
                terms = supervised_terms(forward(model, item, device),
                                         torch.tensor([item["target"]], device=device), item["primitive"], a.rps_weight)
                # FSDP averages ranks on EVERY backward; dummy decisions carry zero weight.
                weight = 0.0 if index is None else world / real
                (terms["loss"] * weight).backward()
                if index is not None:
                    sums += torch.stack([terms[k].detach().double() for k in ("loss", "ce", "rps", "kl")])
            norm = model.clip_grad_norm_(1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite FSDP gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            epoch_end = bi + 1 == batches
            eval_epoch = epoch
            epoch, cursor = (epoch + 1, 0) if epoch_end else (epoch, bi + 1)
            stop = step >= cap
            if step % a.log_every == 0 or epoch_end or stop:
                dist.all_reduce(sums)
                emit("update", global_step=step, epoch=eval_epoch, next_batch=cursor,
                     real_decisions=real, terms=dict(zip(("loss", "ce", "rps", "kl"), (sums / real).tolist())),
                     lr=lr, grad_norm=norm.item(), seconds=time.perf_counter() - started)
            if step % a.eval_every == 0 or epoch_end or stop:
                evaluate(eval_epoch, "epoch_end" if epoch_end else "stop" if stop else "periodic")
            if step % a.save_every == 0 or epoch_end or stop:
                save_checkpoint()
            if stop:
                break
    summary = dict(global_step=step, epoch=epoch, next_batch=cursor, total_steps=total_steps,
                   completed=step == total_steps, capped=step < total_steps, best=best, config=config)
    rank_zero(lambda: atomic_json(output / "summary.json", summary))
    emit("finished", **summary)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "train-data", "dev-data", "output-dir"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--resume", help="Completed generation directory or output-dir/latest.json")
    for name, default in (("epochs", 3), ("global-batch", 32), ("seed", 42), ("max-length", 4096),
                          ("eval-every", 100), ("log-every", 10), ("save-every", 100),
                          ("max-updates", 0), ("train-limit", 0), ("dev-limit", 0)):
        ap.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 2.5e-5), ("min-lr", 1e-6), ("weight-decay", .01),
                          ("warmup-ratio", .05), ("rps-weight", .25)):
        ap.add_argument("--" + name, type=float, default=default)
    a = ap.parse_args()
    if (min(a.epochs, a.global_batch, a.max_length, a.eval_every, a.log_every, a.save_every) < 1
            or min(a.max_updates, a.train_limit, a.dev_limit) < 0
            or not all(math.isfinite(x) for x in (a.learning_rate, a.min_lr, a.weight_decay, a.warmup_ratio, a.rps_weight))
            or not 0 <= a.min_lr <= a.learning_rate or a.learning_rate <= 0
            or min(a.weight_decay, a.rps_weight) < 0 or not 0 <= a.warmup_ratio < 1):
        ap.error("Invalid count/optimizer/scheduler settings")
    for key in ("model_path", "train_data", "dev_data", "output_dir", "resume"):
        if getattr(a, key):
            setattr(a, key, str(Path(getattr(a, key)).expanduser().resolve()))
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    started, stream = time.perf_counter(), None

    def emit(event, **values):
        nonlocal stream
        if stream is None:
            stream = open(Path(a.output_dir) / ("rank%d.events.jsonl" % rank), "a", encoding="utf-8")
        row = dict(event=event, rank=rank, elapsed_seconds=time.perf_counter() - started, memory=memory(), **values)
        text = json.dumps(row, allow_nan=False)
        stream.write(text + "\n")
        stream.flush()
        print(text, flush=True)
    try:
        run(a, device, emit)
    finally:
        if stream is not None:
            stream.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
