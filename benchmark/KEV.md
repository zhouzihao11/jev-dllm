# Native Kev reproduction

Kev is an optional native quality baseline on the fixed 6,744-decision Core,
not a replacement for the released DLLM/Laya renderer. This adapter migrates
research implementation `138df54a2daf541678f34e7b422121e5ad7a2e30` without
changing its model math. The release migration receives source/static review
only: no server access, new inference, installation or fresh-environment test.
The research run completed all 6,744 attempts (6,523 successful, 221 explicit
overlength rejections) and a 51-row smoke/native parity comparison with maximum
probability difference zero. Those are research results, not a release rerun.

## Pinned external assets

| Asset | Upstream | Revision |
|---|---|---|
| Native code | `jaredpalmer/kev` on GitHub | `3e1cd3bb588a388a06827443380befece23e68c7` |
| Adapter and head | `jaredpalmer/kev-0.8b` on Hugging Face | `9a45d25eb2ab761841196625383fa1dff0e56c1e` |
| Base | `Qwen/Qwen3.5-0.8B-Base` on Hugging Face | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |

Run acquisition from the repository root, with a resource directory outside
the checkout. This command is for the user to run; it downloads assets but
does not load models, convert weights, install packages or acquire raw training
data. No weights or caches belong in Git.

```bash
KEV_ROOT="$HOME/model-resources/kev-pinned"
python scripts/prepare_kev.py --output-dir "$KEV_ROOT"
```

`--output-dir` is required and must be new or empty. `--cache-dir` defaults to
`~/.cache/jev-dllm/kev`, outside the project. If preparation fails, move the
incomplete output aside and retry with the same cache; incomplete source caches
must also be moved aside. Do not reuse a cache modified with unrelated assets.
The script fetches the exact GitHub tarball, safely extracts only `kev/__init__.py`,
`api.py`, `checkpoint.py`, `model.py`, and upstream LICENSE/README (NOTICE if
present), and preserves their bytes and copyright. There is no serving import.
These four modules are sufficient for this explicit non-serving torch path;
the subset is not a general Kev training/serving installation.

HF files use explicit filenames and exact revisions, not CLI include patterns.
The base includes config, tokenizer, vocab/merges, preprocessing metadata,
license/card, index, and **every weight shard named by that index**. The local
pinned index inspected for this migration names one shard
(`model.safetensors-00001-of-00001.safetensors`), not three. Adapter files are
`adapter_config.json`, `adapter_model.safetensors`, `head.pt`, `tokenizer.json`,
`tokenizer_config.json`, `provenance.json`, `training_config.json`, `README.md`.
If `huggingface_hub` is available, acquisition uses its cache; bare Python has a
standard-library download fallback. `HF_TOKEN` is read from the environment,
never written into the manifest or URLs; the shared HTTP redirect helper strips
authorization on cross-origin redirects. No token is printed in error messages.

The result is portable: `source/kev/`, `source/LICENSE`, `models/base/`,
`models/kev-0.8b/`, and `kev_paths.json`. Its path values are relative to
`KEV_ROOT`, and source IDs/revisions and file lists record acquisition provenance,
not a cryptographic audit or an automatic evaluator configuration. Relay this
whole directory to an execution machine if needed, and update only `KEV_ROOT`.
The evaluator does not download or rewrite these assets. It records actual
import origins and checks available base-config name/revision metadata; missing
revision metadata is explicitly `not_recorded_in_config`, not proof of weight
identity. Review external code and upstream terms before executing it; see
[third-party attribution](../THIRD_PARTY.md).

## Isolated Python runtime

Keep the existing prm/main environment and main requirements unchanged.
Only Kev inference uses a separate Python 3.12 environment; scoring remains
in the original parent environment with its existing metric dependencies.
The optional requirements pin the versions used in the successful research
environment, not all transitive dependencies or a cross-platform lock.
Research installation succeeded using reachable mirrors; a clean public-index
installation has not been validated by this migration.

Example setup commands for the user, not actions performed during migration:

```bash
conda create -n kev-eval python=3.12
PYTHONNOUSERSITE=1 conda run -n kev-eval python -m pip install -r benchmark/requirements-kev.txt
KEV_PY="$(conda run -n kev-eval python -c 'import sys; print(sys.executable)')"
```

The research runtime used Torch 2.8.0 with CUDA 12.8. The default Linux CUDA
wheel needs a compatible NVIDIA driver (570-series or newer for CUDA 12.8);
select an official Torch wheel appropriate for the execution platform rather
than assuming every platform has the same CUDA build. No optional optimized
DeltaNet/causal-convolution kernels are required or claimed. The native FP32
PyTorch reference fallback can be slow. This release introduces no JIT kernel,
fused path, precision fallback or rewritten weight format. Do not pip-install
the upstream project and its unrelated serving/training extras.

## Evaluate with the original parent Python

Set `PRM_PY` to the original scoring interpreter, `CORE_6744` to the prepared
fixed Core, and `OUTPUT` to a fresh directory under an existing parent.
`KEV_PY` is the isolated child executable, never the parent interpreter.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PRM_PY" benchmark/evaluate.py \
  --data-root "$CORE_6744" --output-dir "$OUTPUT" \
  --backend kev --backend-python "$KEV_PY" \
  --kev-source "$KEV_ROOT/source" --base-path "$KEV_ROOT/models/base" \
  --model-path "$KEV_ROOT/models/kev-0.8b" \
  --device cuda:0 --dtype float32 --batch-size 1 --max-length 4096 \
  --smoke --smoke-limit 1
```

For full evaluation choose another new output directory and remove `--smoke
--smoke-limit 1`. `--predict-only` omits scoring. The child automatically sets
`PYTHONNOUSERSITE=1` and removes `PYTHONPATH`; other variables, including offline
flags, are inherited. The scorer uses the parent's `sys.executable` in another
process and retains its original environment. Input inference opens only the
bundle manifest and inputs, not labels. Actual gold is accessed separately by
the scorer; this is data-flow isolation, not an OS sandbox for external code.

After full evaluation **with scoring**, summarize its existing artifacts using
any standard-library Python (no Kev environment, models or data needed):

```bash
python benchmark/summarize_kev.py --core-run "$CORE_OUTPUT" \
  --output-dir "$HOME/benchmark-results/kev-core-summary"
```

`CORE_OUTPUT` is the full evaluation output, containing `run.json` and
`scores/report.json`, not the `scores/` directory. The summary output must be
new under an existing parent and outside Git. It produces `REPORT.md` and
`summary.json`: Core14 per-suite accuracy/ECE15/NLL/Brier, coverage, ordinal
MAE/RPS target denominators, and equal-suite Nimble9 diagnostic means. The exact
17 suite names and retained/planned counts must match `manifests/core_v1.json`
(6,744 attempts). MultiNLI and Aegis2 training sources and PAWS development
exposure are excluded, not recast as certified unseen data.

Metrics are copied at scorer precision (hard metrics rounded to four decimals).
Aggregate means and weighted accuracy use those stored values, not exact
correct-count reconstruction; mean ECE is not pooled ECE. The report explicitly
uses **Kev native success denominator; not five-model common set**. Historical
ContractNLI Kev alone has 1,853/2,074 successes, versus 1,819/2,074 in the
published five-model common comparison. Reproducing that exact common set
requires joining successful prediction IDs from all five runs with target labels
and rescoring the intersection; this artifact summarizer does not do that.
Run conditions, including actual recorded temperature and runtime options, are
preserved without certifying checkpoint identity. Unknown temperature stays
unknown; overrides are diagnostics, not a claim of native checkpoint calibration.

Kev requires the three path flags plus explicit `--dtype float32` in the direct
CLI (the inherited global dtype default remains BF16). Batch size defaults to
one and any other Kev batch size is rejected; no silent batch-32 serial loop.
Matrix model entries support `backend_python`, `kev_source`, `base_path`, and
optional `kev_temperature`; the matrix chooses FP32 for Kev if dtype is omitted:

```json
[
  {
    "name": "kev-0_8b",
    "backend": "kev",
    "model_path": "/path/to/kev-pinned/models/kev-0.8b",
    "backend_python": "/path/to/kev-eval/bin/python",
    "kev_source": "/path/to/kev-pinned/source",
    "base_path": "/path/to/kev-pinned/models/base",
    "dtype": "float32",
    "device": "cuda:0"
  }
]
```

Save the runtime model list outside the repository, then use the original Python:

```bash
"$PRM_PY" benchmark/evaluate_matrix.py --data-root "$CORE_6744" \
  --models "$KEV_MODELS_JSON" --output-dir "$MATRIX_OUTPUT" \
  --batch-size 1 --max-length 4096
```

The matrix's global batch default is one. A mixed Kev matrix with global
`--batch-size 32` is rejected before creating output. Separate Kev batch-one
and DLLM/Laya batch-32 runs are advisable; they are not throughput-matched.

## Legacy typed decisions

The published legacy typed comparison is a separate 400-case, 2,000-decision
fixture, **not the audited Core**. Package just this fixture with the original
parent Python and its existing research conversion dependencies (including
pandas and a Parquet engine). No private frozen profile, Hub access, other five
external suites, model download or new dataset copy is needed. The actual public
`datasets/manifest.json` entry is `bench/typed_decisions_test.parquet`, not gzip
JSONL. The default reads that already bundled file directly.

From the release repository root, ensure `local_data/` exists first (the output's
parent must exist, and the output directory itself must be new):

```bash
mkdir -p local_data
python scripts/prepare_typed_decisions.py --output-dir "$PWD/local_data/typed-v1"
"$PRM_PY" benchmark/evaluate.py \
  --data-root "$PWD/local_data/typed-v1" --output-dir "$TYPED_OUTPUT" \
  --backend kev --backend-python "$KEV_PY" \
  --kev-source "$KEV_ROOT/source" --base-path "$KEV_ROOT/models/base" \
  --model-path "$KEV_ROOT/models/kev-0.8b" \
  --device cuda:0 --dtype float32 --batch-size 1 --max-length 4096
```

Use a fresh `TYPED_OUTPUT` under an existing parent. The same evaluation flags,
including optional `--smoke --smoke-limit 1`, apply; a smoke prefix is not the
full typed comparison. `--typed-data /path/to/bench/typed_decisions_test.parquet`
can instead select the fixture produced by `scripts/prepare_datasets.py`.
Explicit raw 400-case JSONL is also accepted: it must contain the same `state`,
JSON-string `questions`, JSON-string `gold`, and `workflow` columns in original
order, not 2,000 already flattened decision rows. It is converted to temporary
Parquet solely to call the inherited builder; scratch files are removed and no
dataset is added to Git. Explicit input records its declared public fixture
origin, not a claim of independently verified source identity.

Optionally append the fully scored, single-suite 2,000-decision typed run:

```bash
python benchmark/summarize_kev.py --core-run "$CORE_OUTPUT" \
  --typed-run "$TYPED_OUTPUT" \
  --output-dir "$HOME/benchmark-results/kev-core-and-typed-summary"
```

This adds a separately labelled legacy row (15 table rows total), never a
Core-plus-typed global aggregate. It requires `legacy_typed_decisions_v1`;
historical six-suite external runs are not accepted by this minimal reader.
These summary commands are reproduction instructions, not executed release
measurements or fresh quality validation.

The script uses `bench_local.build_typed_decisions` and
`bench_laya_aligned.external_rows`, preserving historical IDs
`typed_decisions/<case_idx>/<qid>`, 400 intact five-question groups, workflows,
labels, candidate order, reference probabilities and scalar expected scores.
Only absent `criteria` becomes structural JSON null; existing noul dictionaries
are preserved. Inputs contain only `id`, `suite`, `state`, `qdef`; targets and
provenance are separate. The manifest has profile `legacy_typed_decisions_v1`,
scope `legacy_typed_decisions_not_audited_core` and historical source profile
`full_test_v1_20260923_r2`. Its `audit.decisions` file records all 2,000 as retain
for legacy membership, **not a new exposure/source-isolation audit**.

Reference distributions are labelled `reference-distribution` with six-decimal
precision, not human votes or an exact mechanism. Validation allows the bounded
rounding mass error `max(1e-6, K * 0.5 * 10^-6 + 1e-12)` only for explicit legacy
reference scopes. Targets are never renormalized. Core preparation, source-profile
validation and human-vote/exact-mechanism tolerances remain unchanged. There is
no row dropping, quantile filter or unsupported-row exclusion from attempted
coverage. Historical artifact rescoring remains Core-only; run new input-only
evaluation for this bundle.

This integration receives source/static review only, not a release runtime or
quality rerun. Fixed legacy inputs enable comparison with the old input protocol;
they do not make native Kev and published metrics identical protocols. The
release's DLLM/Laya rendering remains exactly its inherited baseline, including
its older noul behavior. Existing dataset licenses and attribution in
`datasets/SOURCES.md` and `THIRD_PARTY.md` continue to apply.

## Native protocol and runtime evidence

- Official `SystemOneRequest` / `to_record`, checkpoint loader, tokenizer,
  LoRA, encode, causal row path and pointer head are used unchanged. Only the
  in-memory base path/revision are redirected to the local snapshot; no config,
  `head.pt` or model file is rewritten. No FastAPI, upstream serving process,
  predictors, dataset module or trainer is imported.
- Default quality protocol is FP32, batch one, strict 4096 for state/branch and
  packed length, torch backend, SDPA, merge requested, LoRA scale one, CUDA graphs
  and fused serving off. Native loader merge rules remain authoritative. CUDA
  TF32, flash SDPA and memory-efficient SDPA are disabled. Inputs contain only
  `state` and `qdef`, one question named `decision`; the native unused constant
  label zero is not the benchmark gold.
- Choice insertion order and score index order are preserved. Native noul keys
  are `false,true`, rendered by Kev as `no,yes`; this does **not** migrate the
  release's older DLLM noul renderer/readout. There are no rotations or date
  facts. `model.forward(enc)` and native softmax produce unrounded FP32 values,
  not the serving/prefix-cache `model.probs` path or rounded API responses.
- The pinned checkpoint temperature is approximately 2.3511 (research runtime
  `2.3510958125672174`). The adapter reads it rather than hard-coding it. An
  explicit positive `--kev-temperature 1` is a separately labelled diagnostic,
  never a temperature fit on Core. Native masks/readout and model math remain
  unchanged. Only `ContextOverflow` is recorded as `unsupported_length` with
  full untruncated token count and null probabilities; OOM/schema errors are fatal.
- `inference.json` and `run.json` record actual interpreter, imported module
  paths, base metadata, head dimensions, dtype, backend, option isolation/hybrid,
  checkpoint/served temperature and load/context options. Predictions retain
  `input_metadata.option_keys`, calibrated logits and inference temperature.
  These are output diagnostics, not inputs to inference.
- Device-synchronized timing covers native forward including the head, softmax
  and CPU tensor copies, excluding encoding/loading/JSON IO. Use the Kev runtime
  timing scope rather than the inherited scorer's generic timing description.
  No causal speed or batch-efficiency advantage is claimed.

## Comparison boundaries

Known upstream training-source contact includes MultiNLI and Aegis2; PAWS has
development exposure. The fixed Core is not purged or certified unseen. The
research record's available exact-state check does not exclude private data,
near duplicates, translations or pretraining contact. Preserve these warnings
and coverage denominators with any quality result; see
[comparison disclosures](../docs/kev_comparison.md).

Kev uses its checkpoint development calibration and a native single-question
protocol, unlike the inherited aligned T=1 comparison. Training sources, model
size, precision and batching differ. The historical DLLM/Laya/S0/S1 comparison
keeps the release's older noul renderer, not the latest research experiment.
Source/static migration review alone cannot establish runtime parity on another
platform, installation compatibility, identical timing scope or new quality
measurements.
