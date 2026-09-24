"""Candidate-only DDP fine tuning of the Qwen masked language model.

All inputs are local files.  Run with ``torchrun --nproc_per_node=2``; the
training forward calls the transformer backbone and projects only A-Z rows (or
shared Yes/No rows) of the language-model head.  The full vocabulary is used
only by the one-item equivalence smoke check.
"""
import argparse
import contextlib
import json
import math
import os
import random
import string

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForMaskedLM, AutoTokenizer

from laya.common import QTYPES, proper_reward, render_options, serialize_state


MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
MASK_ID = 151669
PREFIX = "Decision: "
GROUP_SIZE = 4


def internal(q):
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {x: None for x in crit}
    return {"t": q["type"], "ins": q["instructions"], "crit": crit}


def decision_input(tok, state, qid, qdef, max_length):
    options = render_options(internal(qdef))
    if not 1 <= len(options) <= 26:
        raise ValueError("Expected 1-26 options for question %s" % qid)
    labels = list(string.ascii_uppercase[:len(options)])
    encoded = [tok.encode(x, add_special_tokens=False) for x in labels]
    if any(len(x) != 1 for x in encoded):
        raise ValueError("Every candidate identifier must encode as one token")
    candidates = [x[0] for x in encoded]
    if len(set(candidates)) != len(candidates) or any(x in tok.all_special_ids for x in candidates):
        raise ValueError("Candidate token IDs must be unique and non-special")
    user = (
        "Evaluate this decision using the full state and original question below. "
        "Choose exactly one option, and fill the decision with only its letter identifier.\n\n"
        "State:\n%s\n\nQuestion ID: %s\nType: %s\nInstructions:\n%s\n"
        "Original criteria (original order):\n%s\n\nOptions (original label order):\n%s"
        % (serialize_state(state), qid, qdef["type"], internal(qdef)["ins"],
           json.dumps(qdef.get("criteria"), ensure_ascii=False),
           "\n".join("%s: %s" % x for x in zip(labels, options))))
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) + PREFIX
    ids = tok.encode(prompt, add_special_tokens=False, truncation=False) + [MASK_ID]
    if ids.count(MASK_ID) != 1:
        raise ValueError("Question %s must contain exactly one mask" % qid)
    if len(ids) > max_length:
        raise ValueError("Question %s exceeds --max-length %d" % (qid, max_length))
    return ids, candidates, labels, options, prompt


def target_for(q, gold):
    crit = q.get("criteria", {})
    if q["type"] == "choice":
        keys = list(crit.keys())
    elif q["type"] == "noul":
        keys = ["false", "true"]
    elif q["type"] == "score":
        keys = [str(i) for i in range(len(crit) if isinstance(crit, list) else 4)]
    else:
        raise ValueError("unknown question type: %s" % q["type"])
    values = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
    total = sum(values)
    return [x / total for x in values] if total > 0 else [1.0 / len(values)] * len(values)


def load_items(path, tok, max_length, scoring_mode="identifier", yesno_tokens=None):
    if scoring_mode == "shared_yesno":
        from bench_diff_yesno import decision_input as yesno_input
        if yesno_tokens is None:
            raise ValueError("shared_yesno requires the evaluator token mapping")
        mapping, mask_id, mask_text = yesno_tokens
    rows = pd.read_parquet(path)
    if not 1100 <= len(rows) <= 1300:
        raise ValueError("expected about 1,200 parquet rows, got %d" % len(rows))
    items = []
    for row in rows.to_dict("records"):
        parse = lambda x: json.loads(x) if isinstance(x, str) else x
        state, questions, gold = parse(row["state"]), parse(row["questions"]), parse(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                continue
            if scoring_mode == "shared_yesno":
                ids, positions, options, prompt = yesno_input(tok, state, q, mask_id, mask_text, max_length)
                candidates = [mapping["Yes"], mapping["No"]]
                labels = options
            else:
                ids, candidates, labels, options, prompt = decision_input(tok, state, qid, q, max_length)
            target = target_for(q, gold[qid])
            if len(target) != (len(options) if scoring_mode == "shared_yesno" else len(candidates)):
                raise ValueError("gold/options mismatch for %s" % qid)
            items.append({"ids": ids, "candidates": candidates, "target": target,
                          "qtype": QTYPES[q["type"]], "labels": labels, "options": options,
                          "prompt": prompt})
            if scoring_mode == "shared_yesno":
                items[-1].update(positions=positions, noul=q["type"] == "noul")
    if not 5000 <= len(items) <= 7000:
        raise ValueError("expected about 6,000 training items, got %d" % len(items))
    return items, len(rows)


def yesno_projection(base, hidden, candidate_ids):
    weight = base.lm_head.weight.index_select(0, candidate_ids)
    bias = getattr(base.lm_head, "bias", None)
    if bias is not None:
        bias = bias.index_select(0, candidate_ids)
    if hidden.dtype != weight.dtype:
        raise ValueError("Hidden and lm_head dtypes differ; refusing implicit conversion")
    with torch.autocast("cuda", enabled=False):
        return torch.nn.functional.linear(hidden, weight, bias).float()


class CandidateModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask, candidate_ids, positions=None, noul=False):
        hidden = self.base.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if positions is not None:
            z = yesno_projection(self.base, hidden[0, positions], candidate_ids[0])
            return z[0, [1, 0]].unsqueeze(0) if noul else (z[:, 0] - z[:, 1]).unsqueeze(0)
        h = hidden[:, -1, :].float()
        rows = self.base.lm_head.weight[candidate_ids]
        logits = torch.einsum("bd,bkd->bk", h, rows.float())
        bias = getattr(self.base.lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias[candidate_ids].float()
        return logits


def amp_context(dtype):
    return contextlib.nullcontext() if dtype == torch.float32 else torch.autocast("cuda", dtype=dtype)


def train_supervised(a, dtype, device, rank, world, local_rank):
    import shutil
    import tempfile
    import time
    import uuid

    from bench_diff_yesno import token_mapping
    from shared_yesno_supervised import (load_structured_items, prediction_record,
                                        summarize_predictions, supervised_terms)

    def rank_zero(action):
        error = [None]
        if rank == 0:
            try:
                action()
            except Exception as exc:
                error[0] = "%s: %s" % (type(exc).__name__, exc)
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(error[0])

    def write_json(path, value):
        with open(path, "x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)

    def append_json(name, value):
        with open(os.path.join(output, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(value, allow_nan=False) + "\n")

    output = os.path.realpath(os.path.expanduser(a.output_dir))
    model_path = os.path.realpath(os.path.expanduser(a.model_path))
    resume = os.path.realpath(os.path.expanduser(a.resume)) if a.resume else None
    config = {k: v for k, v in vars(a).items()
              if k not in ("resume", "max_updates", "code_commit", "save_predictions")}
    config.update(model_path=model_path, output_dir=output, world_size=world)
    for key in ("train_data", "dev_data"):
        path = os.path.realpath(os.path.expanduser(getattr(a, key)))
        stat = os.stat(path)
        config[key] = {"path": path, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if resume:
        if os.path.dirname(resume) != output:
            raise ValueError("--resume must be a complete checkpoint inside this output-dir")
        with open(os.path.join(output, "run.json")) as f:
            run = json.load(f)
        with open(os.path.join(resume, "training_metadata.json")) as f:
            checkpoint_meta = json.load(f)
        if checkpoint_meta["run_id"] != run["run_id"]:
            raise ValueError("Resume checkpoint belongs to another run")
    else:
        run = {"run_id": str(uuid.uuid4()) if rank == 0 else None}
        payload = [run]
        dist.broadcast_object_list(payload, src=0)
        run = payload[0]

    def prepare_output():
        if not resume and os.path.exists(output) and os.listdir(output):
            raise FileExistsError("refusing to overwrite non-empty output-dir: %s" % output)
        os.makedirs(output, exist_ok=True)
    rank_zero(prepare_output)

    load_path = resume or model_path
    tok = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, local_files_only=True)
    base = AutoModelForMaskedLM.from_pretrained(
        load_path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True)
    yesno_tokens = token_mapping(tok, base)
    items = load_structured_items(config["train_data"]["path"], tok, a.max_length,
                                  yesno_tokens, limit=a.train_limit)
    dev = load_structured_items(config["dev_data"]["path"], tok, a.max_length,
                                yesno_tokens, limit=a.dev_limit)
    if any(item.get("split") == "test" for item in items + dev):
        raise ValueError("Test data is not accepted by the trainer; evaluate heldout data separately")
    counts = [None] * world
    dist.all_gather_object(counts, (len(items), len(dev), len(items[rank::world])))
    if not items or not dev or len(set(counts)) != 1 or len(items) % world:
        raise ValueError("Nonempty train/dev and equal per-rank train counts required; got %r" % counts)
    batches = math.ceil((len(items) // world) / a.micro_batch)
    total_steps = math.ceil(batches / a.grad_accum) * a.epochs
    warmup_steps = math.ceil(total_steps * a.warmup_ratio)
    config.update(train_count=len(items), dev_count=len(dev), rank_counts=counts,
                  total_steps=total_steps, warmup_steps=warmup_steps,
                  token_mapping=yesno_tokens[0], mask_id=yesno_tokens[1])
    # JSON round-trip keeps rank-count tuples comparable to saved runtime metadata.
    config = json.loads(json.dumps(config))
    if resume:
        if run["config"] != config or checkpoint_meta["config"] != config:
            raise ValueError("Resume configuration/data metadata differs from the saved run")
    else:
        run.update(config=config, code_commit=a.code_commit)
        rank_zero(lambda: write_json(os.path.join(output, "run.json"), run))

    base.config.use_cache = False
    base.model.config.use_cache = False
    base.requires_grad_(True)
    base.to(device)
    try:
        base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        base.model.gradient_checkpointing_enable()
    if a.check_projection:
        base.eval()
        with torch.no_grad(), amp_context(dtype):
            item = items[0]
            ids = torch.tensor([item["ids"]], device=device)
            attention = torch.ones_like(ids, dtype=torch.bool)
            candidate_ids = torch.tensor(item["candidates"], device=device)
            full = base(input_ids=ids, attention_mask=attention).logits[0, item["positions"]]
            full = full.index_select(-1, candidate_ids).float()
            hidden = base.model(input_ids=ids, attention_mask=attention).last_hidden_state[0, item["positions"]]
            projected = yesno_projection(base, hidden, candidate_ids)
            if not torch.isfinite(full).all() or not torch.isfinite(projected).all():
                raise FloatingPointError("Non-finite projection equivalence logits")
            print(json.dumps(dict(event="projection_check", rank=rank,
                                  max_abs_diff=(full - projected).abs().max().item())), flush=True)
            if not torch.allclose(full, projected, rtol=0.02, atol=0.08):
                raise RuntimeError("candidate projection disagrees with full logits")
        del ids, attention, candidate_ids, full, hidden, projected
        base.train()
    wrapper = DDP(CandidateModel(base), device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)

    def lr_factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        floor = a.min_lr / a.learning_rate
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp_dtype == "float16")
    step, start_epoch, next_batch, elapsed_before = 0, 0, 0, 0.0
    best = None
    if resume:
        state = torch.load(os.path.join(resume, "trainer_state.pt"), map_location="cpu", weights_only=False)
        if state["run_id"] != run["run_id"] or state["config"] != config:
            raise ValueError("Optimizer checkpoint configuration does not match this run")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, start_epoch, next_batch = state["step"], state["epoch"], state["next_batch"]
        if (not 0 <= start_epoch <= a.epochs or not 0 <= next_batch < batches
                or next_batch % a.grad_accum or (start_epoch == a.epochs and next_batch)
                or step != start_epoch * math.ceil(batches / a.grad_accum) + next_batch // a.grad_accum
                or scheduler.last_epoch != step or len(state["rng"]) != world):
            raise ValueError("Invalid optimizer-boundary resume position or schedule")
        best, elapsed_before = state["best"], state["elapsed_seconds"]
        # A recovered previous checkpoint may predate a successfully published best.
        best_path = os.path.join(output, "best", "training_metadata.json")
        if os.path.isfile(best_path):
            with open(best_path) as f:
                saved_best = json.load(f)
            if saved_best["run_id"] != run["run_id"]:
                raise ValueError("Best checkpoint belongs to another run")
            if best is None or saved_best["best"]["kl"] < best["kl"]:
                best = saved_best["best"]
        elif best is not None:
            raise FileNotFoundError("Saved best is missing; recover the retained best directory before resuming")
        rng = state["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        del state
    started = time.monotonic()
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def metadata(epoch):
        return dict(run_id=run["run_id"], config=config, code_commit=a.code_commit,
                    step=step, epoch=epoch, best=best, elapsed_seconds=elapsed(),
                    max_updates=a.max_updates, candidate_only=True, temperature=1.0,
                    prompt_protocol="bench_diff_yesno.py", resume_from=resume)

    def owned(path):
        if os.path.islink(path):
            raise ValueError("Refusing to replace checkpoint symlink: %s" % path)
        with open(os.path.join(path, "training_metadata.json")) as f:
            if json.load(f)["run_id"] != run["run_id"]:
                raise ValueError("Refusing to replace another run's checkpoint: %s" % path)

    def save_checkpoint(name, epoch, next_bi=0, complete=False):
        rngs = None
        if complete:
            rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                       torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
            rngs = [None] * world
            dist.all_gather_object(rngs, rng)

        def save():
            target = os.path.join(output, name)
            if os.path.exists(target):
                owned(target)
                if name not in ("best", "resume-latest"):
                    target += "-recovery-" + uuid.uuid4().hex
            temp = tempfile.mkdtemp(prefix=".%s-new-" % name, dir=output)
            wrapper.module.base.save_pretrained(temp, safe_serialization=True)
            tok.save_pretrained(temp)
            write_json(os.path.join(temp, "training_metadata.json"), metadata(epoch))
            if complete:
                torch.save(dict(run_id=run["run_id"], config=config, step=step, epoch=epoch,
                                next_batch=next_bi, optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                                rng=rngs, best=best, elapsed_seconds=elapsed()),
                           os.path.join(temp, "trainer_state.pt"))
            retired = None
            if name == "resume-latest" and os.path.exists(target):
                previous = os.path.join(output, "resume-previous")
                if os.path.exists(previous):
                    owned(previous)
                    retired = os.path.join(output, ".resume-retired-" + uuid.uuid4().hex)
                    os.rename(previous, retired)
                os.rename(target, previous)
            elif os.path.exists(target):
                retired = os.path.join(output, ".best-retired-" + uuid.uuid4().hex)
                os.rename(target, retired)
            os.rename(temp, target)
            if retired:
                shutil.rmtree(retired)
        rank_zero(save)

    def forward(model, item):
        ids = torch.tensor([item["ids"]], device=device)
        candidates = torch.tensor([item["candidates"]], device=device)
        logits = model(ids, torch.ones_like(ids, dtype=torch.bool), candidates,
                       item["positions"], item["noul"]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite supervised logits")
        return logits

    def evaluate(epoch, reason):
        nonlocal best
        wrapper.eval()
        records = []
        with torch.no_grad(), amp_context(dtype):
            for item in dev[rank::world]:
                records.append(prediction_record(item, forward(wrapper.module, item)))
        shards = [None] * world if rank == 0 else None
        dist.gather_object(records, shards, dst=0)
        result = [None]

        def publish():
            all_records = [record for shard in shards for record in shard]
            metrics = summarize_predictions(all_records)
            if not math.isfinite(metrics["kl"]):
                raise FloatingPointError("Non-finite mean dev KL")
            result[0] = metrics["kl"]
            event = dict(step=step, epoch=epoch, reason=reason, metrics=metrics,
                         elapsed_seconds=elapsed())
            event_name = "dev-step-%d-%s" % (step, uuid.uuid4().hex)
            write_json(os.path.join(output, event_name + ".json"), event)
            if a.save_predictions:
                with open(os.path.join(output, event_name + "-predictions.jsonl"), "x") as f:
                    for record in all_records:
                        f.write(json.dumps(record, allow_nan=False) + "\n")
            append_json("metrics_history.jsonl", event)
            print(json.dumps(dict(event="dev", step=step, epoch=epoch, reason=reason,
                                  count=metrics["count"], kl=metrics["kl"], ce=metrics["ce"],
                                  hard_accuracy=metrics["hard_accuracy"], rps=metrics["rps"],
                                  elapsed_seconds=event["elapsed_seconds"],
                                  metrics_file=event_name + ".json"), allow_nan=False), flush=True)
        rank_zero(publish)
        dist.broadcast_object_list(result, src=0)
        if best is None or result[0] < best["kl"]:
            best = dict(step=step, epoch=epoch, kl=result[0])
            save_checkpoint("best", epoch)
        wrapper.train()

    rank_zero(lambda: print(json.dumps(dict(event="start", step=step, total_steps=total_steps,
                                            config=config, max_updates=a.max_updates)), flush=True))
    if not resume:
        evaluate(0, "initial")
    cap = min(total_steps, a.max_updates) if a.max_updates else total_steps
    if step >= cap:
        evaluate(start_epoch, "stop")
        save_checkpoint("resume-latest", start_epoch, next_batch, complete=True)
    else:
        for epoch in range(start_epoch, a.epochs):
            indices = list(range(len(items)))
            random.Random(a.seed + epoch).shuffle(indices)
            mine = indices[rank::world]
            begin = next_batch if epoch == start_epoch else 0
            for bi in range(begin, batches, a.grad_accum):
                end_bi = min(batches, bi + a.grad_accum)
                window = mine[bi * a.micro_batch:end_bi * a.micro_batch]
                sums = torch.zeros(4, dtype=torch.float64, device=device)
                lr_used = optimizer.param_groups[0]["lr"]
                for i, index in enumerate(window):
                    item = items[index]
                    sync = contextlib.nullcontext() if i == len(window) - 1 else wrapper.no_sync()
                    # no_sync must enclose both forward and backward, including checkpoint recomputation.
                    with sync:
                        with amp_context(dtype):
                            logits = forward(wrapper, item)
                            target = torch.tensor([item["target"]], dtype=torch.float32, device=device)
                            terms = supervised_terms(logits, target, item["primitive"], a.rps_weight)
                            loss = terms["loss"] / len(window)
                        if not all(torch.isfinite(value).all() for value in terms.values()):
                            raise FloatingPointError("Non-finite supervised loss/metrics")
                        scaler.scale(loss).backward()
                    sums += torch.stack([terms[k].detach().double() for k in ("loss", "ce", "rps", "kl")])
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient norm")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                epoch_end = end_bi == batches
                stop = step >= cap
                if step % 10 == 0 or epoch_end or stop:
                    dist.all_reduce(sums)
                    values = (sums / (len(window) * world)).tolist()
                    progress = dict(zip(("loss", "ce", "rps", "kl"), values))
                    progress.update(step=step, epoch=epoch + 1, next_batch=end_bi, lr=lr_used,
                                    grad_norm=float(norm), elapsed_seconds=elapsed(),
                                    global_window_examples=len(window) * world, total_steps=total_steps)
                    def log_progress():
                        append_json("training_history.jsonl", progress)
                        print(json.dumps(progress, allow_nan=False), flush=True)
                    rank_zero(log_progress)
                if step % a.eval_every == 0 or epoch_end or stop:
                    evaluate(epoch + 1, "epoch_end" if epoch_end else "stop" if stop else "periodic")
                if epoch_end and a.save_epoch_checkpoints:
                    save_checkpoint("epoch-%d" % (epoch + 1), epoch + 1)
                if step % a.save_every == 0 or (epoch_end and a.save_epoch_checkpoints) or stop:
                    save_checkpoint("resume-latest", epoch + 1 if epoch_end else epoch,
                                    0 if epoch_end else end_bi, complete=True)
                if stop:
                    break
            if step >= cap:
                break
    rank_zero(lambda: append_json("training_history.jsonl",
                                 dict(event="finished", step=step, total_steps=total_steps,
                                      capped=step < total_steps, best=best, elapsed_seconds=elapsed())))
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--scoring-mode", choices=("identifier", "shared_yesno"), default="identifier")
    ap.add_argument("--smoke-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss-mode", choices=("rlcd", "supervised"), default="rlcd")
    ap.add_argument("--dev-data")
    ap.add_argument("--learning-rate", type=float, default=2.5e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--rps-weight", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-epoch-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                    help="Supervised: save epoch models and force complete saves at epoch ends")
    ap.add_argument("--check-projection", action="store_true",
                    help="Supervised: check one example against full-vocabulary logits before training")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--code-commit")
    ap.add_argument("--save-predictions", action="store_true")
    a = ap.parse_args()
    if min(a.epochs, a.micro_batch, a.grad_accum, a.max_length) < 1 or a.smoke_batches < 0:
        ap.error("epochs, micro-batch, grad-accum, and max-length must be positive")
    if a.loss_mode == "supervised":
        if a.scoring_mode != "shared_yesno" or not a.dev_data or a.smoke_batches:
            ap.error("supervised requires shared_yesno and --dev-data; use --max-updates, not --smoke-batches")
        if (not all(math.isfinite(v) for v in (a.learning_rate, a.min_lr, a.weight_decay,
                                               a.warmup_ratio, a.rps_weight))
                or not 0 <= a.min_lr <= a.learning_rate or a.learning_rate <= 0
                or a.weight_decay < 0 or a.rps_weight < 0 or not 0 <= a.warmup_ratio < 1
                or min(a.eval_every, a.save_every) < 1
                or min(a.train_limit, a.dev_limit, a.max_updates) < 0):
            ap.error("Invalid supervised optimizer, schedule, interval, or limit arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL DDP")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.amp_dtype]
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = a.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if a.loss_mode == "supervised":
        train_supervised(a, dtype, device, rank, world, local_rank)
        return
    model_path, data_path, output = [os.path.abspath(os.path.expanduser(x))
                                     for x in (a.model_path, a.train_data, a.output_dir)]
    if rank == 0 and os.path.exists(output) and os.listdir(output):
        raise FileExistsError("refusing to overwrite non-empty output-dir: %s" % output)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if a.scoring_mode == "identifier":
        if tok.mask_token_id not in (None, MASK_ID):
            raise ValueError("tokenizer mask ID disagrees with checkpoint")
        if tok.encode(tok.convert_ids_to_tokens(MASK_ID), add_special_tokens=False) != [MASK_ID]:
            raise ValueError("mask ID does not round-trip")
        items, n_rows = load_items(data_path, tok, a.max_length)
    base = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True,
                                                 torch_dtype=dtype, local_files_only=True)
    if a.scoring_mode == "shared_yesno":
        from bench_diff_yesno import token_mapping
        yesno_tokens = token_mapping(tok, base)
        items, n_rows = load_items(data_path, tok, a.max_length, a.scoring_mode, yesno_tokens)
    base.to(device)
    try:
        base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        base.model.gradient_checkpointing_enable()
    base.train()
    if a.scoring_mode == "shared_yesno":
        base.eval()
    # One deliberate full-vocabulary check; no subsequent training forward calls lm_head().
    with torch.no_grad(), amp_context(dtype):
        x = torch.tensor([items[0]["ids"]], device=device)
        m = torch.ones_like(x, dtype=torch.bool)
        if a.scoring_mode == "shared_yesno":
            candidate_ids = torch.tensor(items[0]["candidates"], device=device)
            full = base(input_ids=x, attention_mask=m).logits[0, items[0]["positions"]]
            full = full.index_select(-1, candidate_ids)
            hidden = base.model(input_ids=x, attention_mask=m).last_hidden_state[0, items[0]["positions"]]
            projected = yesno_projection(base, hidden, candidate_ids)
        else:
            full = base(input_ids=x, attention_mask=m).logits[0, -1, items[0]["candidates"]]
            hidden = base.model(input_ids=x, attention_mask=m).last_hidden_state[0, -1].float()
            rows = base.lm_head.weight[torch.tensor(items[0]["candidates"], device=device)].float()
            projected = torch.einsum("d,kd->k", hidden, rows)
            if getattr(base.lm_head, "bias", None) is not None:
                projected = projected + base.lm_head.bias[torch.tensor(items[0]["candidates"], device=device)].float()
        full = full.float()
        projected = projected.float()
        if not torch.allclose(full, projected, rtol=0.02, atol=0.08):
            raise RuntimeError("candidate projection disagrees with full logits")
    if a.scoring_mode == "shared_yesno":
        base.train()
    wrapper = DDP(CandidateModel(base), device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=2.5e-5, weight_decay=0.01)
    mine = items[rank::world]
    batches = math.ceil(len(mine) / a.micro_batch)
    if a.smoke_batches:
        batches = min(batches, a.smoke_batches)
    updates = max(1, math.ceil(batches / a.grad_accum) * a.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp_dtype == "float16")
    if rank == 0:
        os.makedirs(output, exist_ok=True)
        print("Training %d rows / %d items across %d ranks" % (n_rows, len(items), world), flush=True)
    dist.barrier()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(a.epochs):
        random.Random(a.seed + epoch + rank).shuffle(mine)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, a.epochs - 1)
        for bi in range(batches):
            chunk = mine[bi * a.micro_batch:(bi + 1) * a.micro_batch]
            losses = []
            for item in chunk:
                ids = torch.tensor([item["ids"]], device=device)
                candidates = torch.tensor([item["candidates"]], device=device)
                mask = torch.ones((1, len(item["target"])), dtype=torch.bool, device=device)
                target = torch.tensor([item["target"]], dtype=torch.float32, device=device)
                with amp_context(dtype):
                    if a.scoring_mode == "shared_yesno":
                        logits = wrapper(ids, torch.ones_like(ids, dtype=torch.bool), candidates,
                                         item["positions"], item["noul"]).float()
                    else:
                        logits = wrapper(ids, torch.ones_like(ids, dtype=torch.bool), candidates).float()
                eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask.unsqueeze(0), -1e4), -1)
                with torch.no_grad():
                    reward = proper_reward(q, target.unsqueeze(0), torch.tensor([item["qtype"]], device=device),
                                           mask.unsqueeze(0), w_sph=0.75, w_rps=1.0)
                    advantage = (reward - reward.mean(0, keepdim=True)) / (reward.std(unbiased=False) + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2).sum(-1)) / (2 * sigma ** 2)
                ce = -(target * torch.log_softmax(logits, -1)).sum(-1)
                loss = (-(advantage * logp).mean() + ce.mean()) / (a.grad_accum * len(chunk))
                if not torch.isfinite(logits).all() or not torch.isfinite(q).all() or not torch.isfinite(reward).all() or not torch.isfinite(loss):
                    raise FloatingPointError("non-finite logits, probabilities, reward, or loss")
                losses.append(loss)
            for loss in losses:
                scaler.scale(loss).backward()
            if (bi + 1) % a.grad_accum == 0 or bi + 1 == batches:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient norm")
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print("epoch %d/%d complete" % (epoch + 1, a.epochs), flush=True)
    dist.barrier()
    if rank == 0:
        base.eval()
        base.save_pretrained(output, safe_serialization=True)
        tok.save_pretrained(output)
        mapping = (yesno_tokens[0] if a.scoring_mode == "shared_yesno" else
                   {label: candidate for item in items for label, candidate in zip(item["labels"], item["candidates"])})
        prompt_examples = {}
        for item in items:
            prompt_examples.setdefault(str(item["qtype"]), item["prompt"])
        with open(os.path.join(output, "training_metadata.json"), "w") as f:
            json.dump({"candidate_only": True,
                       "mask_id": yesno_tokens[1] if a.scoring_mode == "shared_yesno" else MASK_ID,
                       "scoring_mode": a.scoring_mode, "token_mapping": mapping,
                       "prompt_protocol": "bench_diff_yesno.py" if a.scoring_mode == "shared_yesno" else "bench_masked_typed.py",
                       "prompt_examples": prompt_examples,
                       "amp_dtype": a.amp_dtype, "epochs": a.epochs, "seed": a.seed,
                       "train_rows": n_rows, "train_decisions": len(items), "world_size": world}, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
