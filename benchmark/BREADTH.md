# Breadth-v1 bridge

This is the full original official Breadth-v1 population, not Core or a
decontaminated subset. No training-overlap exclusion is applied. Keep all 14
datasets and five areas. Official test has 1,990 records / 3,089 questions;
development has 1,990 / 3,075. Partition SHA and record/question/source counts
must match the existing official manifest supplied by `--kev-source`.

Use an existing checkout of official Kev (integration inspected against revision
`fc4a17e194bca35a48c57211392ec49ea54a94ac`). `--suite` must already contain the
identical manifest and authorized frozen partition; nothing is downloaded.
Do not publicly redistribute the partition, prepared inputs, metadata or targets.
The official manifest documents source-specific redistribution restrictions.

## Usage

Run from the repository root. Paths below are caller-supplied examples.
`PYTHON` is the existing inference interpreter; `KEV_PYTHON` is the existing
official Kev interpreter, not a new environment. Prepare/export use only the
standard library. Report uses its caller's Python to run the exact supplied
`scripts/breadth_report.py`, including its `kev.metrics` dependencies.

```bash
export KEV_SOURCE=/path/to/pinned/kev
export SUITE=/path/to/authorized/evals/breadth-v1
export BUNDLE=/path/to/new/breadth-inputs
export RUN=/path/to/new/predictions-run
export NATIVE=/path/to/new/native-rows
export PYTHON=/path/to/existing/inference/python
export KEV_PYTHON=/path/to/existing/kev/python

"$PYTHON" benchmark/breadth.py prepare --kev-source "$KEV_SOURCE" \
  --suite "$SUITE" --split test --allow-test --out "$BUNDLE"

"$PYTHON" benchmark/evaluate.py --data-root "$BUNDLE" --backend dream \
  --model-path /path/to/existing/checkpoint --output-dir "$RUN" \
  --batch-size 1 --max-length 4096 --predict-only

"$PYTHON" benchmark/breadth.py export --kev-source "$KEV_SOURCE" \
  --suite "$SUITE" --split test --allow-test --data-root "$BUNDLE" \
  --predictions "$RUN/predictions.jsonl" --backend dream --out "$NATIVE"

"$KEV_PYTHON" benchmark/breadth.py report --kev-source "$KEV_SOURCE" \
  --suite "$SUITE" --split test --allow-test --result Dream="$NATIVE" \
  --result Other=/path/to/other/local/native-run \
  --reference Published=/path/to/published/official/report.json \
  --reference-system Jev Kev-0.8B Kev-4B Kev-9B 'Kev-9B (released)' Kev-27B \
  --out /path/to/new/comparison
```

For the existing fixed S1 diagnostic, add
`--temperature 1.8739716972345604` to export, using a separate output directory.
No temperature is fitted. The override is an absolute temperature applied once
to raw logits; native Kev's already-served logits are first restored using their
recorded native temperature. Without an override, saved probabilities and native
temperature are preserved. For DLLM/Laya/Kev inference, use their existing
`evaluate.py` CLI requirements unchanged. Always use `--predict-only`: the Core
scorer is not the official Breadth scorer. Do not use smoke/suite selection for a
full export; every question needs a saved prediction, including unsupported ones.

## Conventions and outputs

- Inputs contain only `id`, `suite`, `state`, `qdef`, grouped in official dataset
  order with frozen record/question order within each dataset. Choice insertion
  order, state, instructions and criteria are unchanged. An absent official noul
  `criteria` becomes `null`, the existing bundle representation.
- IDs are reversible JSON `[record_id, question_id]` strings. `metadata.jsonl`
  retains frozen IDs, groups and full record metadata; `targets.jsonl` is separate.
  Inference opens neither file. Export rejoins separately loaded frozen records
  and verifies the input bundle against them.
- DLLM/Dream choice logits are Yes-minus-No logodds. Noul probabilities are
  `[false,true]` with logits `[No,Yes]` from raw `[[Yes,No]]`. Laya uses its raw
  option vector. Kev uses `input_metadata.logits` (already divided by native
  temperature) and `inference_temperature`. No prompt, renderer, trainer or
  readout changes are made; Dream's historical generic noul renderer stays intact.
- `rows.json` follows `kev.benchmark.prediction_rows`, including option keys,
  labels, provenance, served logits and temperature. Normalization uses the
  documented official sum tolerance, without flooring probabilities or logits.
  Duplicate, unknown, missing, nonfinite or mismatched prediction IDs/values fail.
  Unsupported predictions are omitted from native rows and itemized in
  `export.json`; their questions remain in the full official scoring denominator.
- Official `report.json`/`report.md` score local rows on the entire frozen test:
  coverage-adjusted accuracy/case-exact, chance-corrected index, all datasets and
  areas, dataset/area/pooled ECE10, Brier and exact logit NLL. The official
  record-clustered paired index bootstrap uses 2,000 resamples, seed 0, for local
  row sets only (including their missing answers as wrong).
- `comparison.json` preserves complete official systems and metrics, adds
  published systems as `TAG:NAME`, and explicitly marks them `official reference`.
  References must have matching test manifest SHA, formula and full panel/counts.
  No paired uncertainty is claimed for published references without local rows;
  the official markdown and bootstrap remain local-only. Optional
  `--reference-system NAME [NAME ...]` selects existing names across supplied
  reference reports only (not local results); requested names missing from all
  reference reports fail. Omit it to include every published system.
- `comparison.md` displays merged JSON values in an overall table, five area
  tables and all 14 dataset tables, with explicit local/reference provenance,
  scoring-unit coverage, question-accuracy versus case-exact semantics, and the
  unchanged local-only bootstrap output. Official `report.json` and `report.md`
  remain unchanged by merging and presentation.

Report inherits the official script's runtime imports and behavior, including its
markdown assumption that overall answered ECE is numeric (an entirely unanswered
system cannot produce that markdown). Compatibility is with the inspected
official schema, not arbitrary future Kev versions. No executable inference or
official report validation is performed as part of this local integration.
