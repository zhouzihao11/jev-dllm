# S1 dataset-only expansion

This builder changes no trainer, loss, renderer, evaluation suite, or S0 input.
Use the original DLLM base and existing trainer separately. Original S0 dev and
mean-KL checkpoint selection remain unchanged. Additional dev is diagnostic only.
No TREC, Laya/S0-long runs, APIs, models, or generated public annotations are used.

## Prepared input contract

All paths are launch-time inputs. `--s0-dir` contains `canonical/train.jsonl`,
`canonical/dev.jsonl` (original 1000 rows), `canonical/test.jsonl`,
`source_manifest.json`, and the original `build_manifest.json` (generation seed).
All S0 splits block additions, even S0 train rows removed by the external audit.

`--source-dir` is the existing prepared S0 public root, with its source manifest,
`prepared/{snli,clinc,sgd}_train.jsonl` and `prepared/sgd_schema.json` (schema may
also be at root). Formats are those consumed by `shared_yesno_sources.py`.
SNLI **must** include original image `group_key`; missing keys fail rather than
falling back to a weaker premise group. Source revisions must match S0.

`--new-source-dir` contains:

- `prepared/dbpedia_train.jsonl`: `{record_id: string, title: string, content: string, label: string}`.
- `prepared/dbpedia_labels.json`: ordered list of all 14 original ontology names.
- `prepared/goemotions_train.jsonl`: `{record_id: string, text: string, labels: list[string]}`; original comment IDs, not export row numbers. Multi-label rows are allowed in the export but discarded before sampling.
- `prepared/goemotions_labels.json`: ordered list of the original 27 emotions plus `neutral`.
- `prepared/boolq_train.jsonl`: `{record_id: string, passage: string, question: string, answer: bool, title?: string}`.
- `source_manifest.json`: `sources.dbpedia`, `sources.goemotions`, and `sources.boolq`, each with nonempty `revision`, `upstream`, and `license` (or `license_declaration`).

Every prepared public file is **original train only**. If `original_split` or
`split` is supplied, it must be `train`; other rows are rejected. No mirrors or
derivatives of LocalLLaMA typed-decisions, AG News, DAIR Emotion, Banking77,
the frozen prompt-injection source, or Stanford SST families are authorized,
regardless of split. Source provenance remains the resource owner's responsibility;
the builder validates declarations, not their factual/legal authenticity. Known
forbidden-family references in declared upstream/dataset/repository fields fail
the build; this lexical check is not a comprehensive mirror-lineage detector.
Converters also verify original IDs and supplied split fields independently of
the loader's train-only filtering.

Release documentation modification: the historical frozen bundle and its
cache-dependent preparation tool are not distributed here. See
`../docs/data_and_models.md` and `../docs/full_eval_profile.template.json`.
`--heldout-profile` must be a locally prepared `full_eval_v1` profile.
All six fixture paths resolve relative to
the profile. Parquet reads project only `state` for typed decisions and `text`
for prompt injection. JSONL text fixtures and Banking77 `text` CSV are supported;
PI JSONL is also accepted when declared. Typed state supports JSON-encoded state
or plain text, matching the inherited reader. All rows, not evaluation prefixes,
are read and checked against declared fixture row counts. Missing, empty,
unsupported, or unreadable fixtures fail closed. External labels, text, matched
example identities, and per-example external errors are never exported or passed
to constructors; only candidate rejection and aggregate audit counts escape.

## Recipe and limitations

The supplied config requests 30,000 added train decisions and 2,000 additional
dev decisions. With the authorized operational ordinal fallback, the intended
train composition is **25,000 public + 5,000 synthetic**, not 28,000 + 2,000.
No independent public score source was locked. The authorized programmatic
operational-rubric fallback is deliberately selected; this is not a claim that
unavailable rating resources were researched and ruled out.
Quotas, seeds, candidate cap, and lexical thresholds are config values. `--seed`
overrides the split/public sampling seed; `synthetic_seed` is independent and must
differ from the original S0 build seed.

Public pools use inherited bounded raw-record reservoir sampling (30,000/source
by default), not all-record prompt expansion. Identity tracking still scales
with the original file; JSONL must be scanned to sample uniformly. GoEmotions
single-label and CLINC domain filtering occur before sampling. Missing/unusable
sources fail. Invalid source rows have rejection tallies, never fabricated labels.

Original groups are shuffled and assigned train/dev proportionally to requested
quotas before decision conversion or content filtering. Only one raw record per
group is selected, so SNLI uses one decision per image group, not duplicate
three-way/binary views. The two SNLI views are mutually exclusive, selected using
the declared quota ratio; binary always asks entailment, never contradiction.
S0 original IDs and inherited image/dialogue/CLINC-text groups block additions.
CLINC uses nonfinancial original intents and K in 2,4,8,16,32,64,120; the inherited
converter's default is unchanged. SGD retains one seeded decision per dialogue,
and an independent annotation check verifies its current active intent and exact
prefix boundary. DBpedia/GoEmotions use full, deterministically permuted original
ontologies. BoolQ groups by normalized passage, joined transitively by nonempty
normalized title, before splitting; this can conservatively reduce availability.

Synthetic generation reuses the inherited `_record`, solver, presentation, and
validator with a different seed. `shared_yesno_s1_synthetic.py` adds a bounded
single pass of fact scenarios in the SAME nine registered families; the S0
generator/defaults are unchanged. `synthetic_candidates_per_pool` controls the
supplement budget (15,000 attempted candidates per pool in the supplied config).
No repeated seeds, retries until quotas fill, repeated views, or new rubrics are
used. Ordinal scenarios vary quantities, prerequisite statuses, version validity,
control limits, test completion, and rollout health, scheduling all rubric levels.
Constraints vary eligibility/cost configurations, allowed/forbidden resource
operations, and original-person ownership histories. Probability scenarios vary
primitive coprime bag vectors, mechanism priors/likelihoods/outcomes, and finite
without-replacement populations/draw counts, using exact known distributions.
Finite bag vectors are reduced by their GCD, not scaled copies of an old vector.

IDs are seed-namespaced; supplements also have a distinct namespace and revision.
For inherited paired groups, a seeded alternating per-family member schedule
balances base/counterfactual or linked views rather than always choosing view 0.
Supplement groups have one factual scenario. Candidate groups are partitioned by
target stratum before selection, and a seeded round-robin over strata retains
family/target coverage without repeating groups. All facts and solver-only query
metadata remain outside model-visible input except the registered observations
and policy presentation. Labels never enter input via metadata.

Synthetic dedup is **exact canonical family plus solver facts**, against every S0
view and all linked facts of selected new groups. Alternative lists are sorted by
name; chronological events retain their order. Fact duplicates are rejected both
while generating supplements and again before output. Public lexical indices do
not reject synthetic candidates because of shared keys/rule words. Synthetic
external screening is normalized exact complete-visible-state matching only (not
isolated policy fields), never approximate JSON-boilerplate similarity.
This is parameter-group splitting,
**not family-heldout**, semantic novelty, or broad new reasoning coverage.
Family and target coverage are reported. Independent deterministic target checks
run in addition to the inherited validator for every accepted synthetic row.
The bounded supplements are designed to support the requested 3,000 ordinal,
1,000 constraint, and 1,000 probability train rows plus dev after exclusions;
achieved counts still require the remote build. Any remaining shortage is reported
without padding or changing the task assumptions.

Dedup uses NFKC/casefold word normalization (punctuation/whitespace folded,
numbers retained), exact original content, and three-word-shingle Jaccard >=0.85
for public texts with at least eight words. Structured content includes the concatenated
original text and individual fields of eight or more words. SGD compares prefix
utterances without its repeated schema. Synthetic facts use the separate exact
handling above. For public candidates, the external firewall additionally checks
full visible state and its long text fields, including policies/schema. The
lexical index uses the rarest available postings first, skips
postings longer than 2,000, and checks at most 256 candidates/query. These bounds
and skip counters are in the report. Exact matching is uncapped; near matching is
approximate and can miss overlaps. No semantic/model-based audit is claimed.
Conservative public lexical matching can reject genuinely different texts.
Public content dedup spans all public S0 and accepted public additions across splits.
Filtered or quota-surplus candidates are not added to the new-content index.

S0 train collisions against full external fixture text are stripped only from
the new train file, with original candidate IDs and reasons in `rejections.jsonl`.
S0 dev is always copied byte-for-byte, even if audit collisions exist; the report
then carries an explicit dev-collision warning for the maintainer to review.
`requires_parent_dev_collision_review=true` and
`heldout_safety=requires_parent_review_of_unchanged_s0_dev` explicitly flag dev
collisions; a completed build is not approval to treat that dev as heldout-safe.
S0 test is audited and blocks additions, but is not exported. This screens the supplied
full fixtures, not unavailable other splits of the forbidden families. It does
not establish absence of base-model pretraining contamination.

The inherited renderer/token mapping is used without truncation and without model
weights. Overlength additions and special-token literals are rejected. All retained
S0 rows/dev/test must render successfully without modification. Reports contain
source/type/K/label/position counts, 256-token length bins, original-record/group
counts, effective quotas, rejected IDs/reasons, audit bounds, and shortages.
One deterministic added train decision is removed if necessary for an even final
2-rank training count; S0 rows are not removed just to fix parity. No padding.

## Remote commands

Run in the existing remote experiment environment with the repo on `PYTHONPATH`
and any existing DLLM stub already configured. No local runtime setup is needed.
Set these shell variables to the resource owner's actual remote paths:

```bash
python "$CODE/research/scripts/build_shared_yesno_s1.py" \
  --s0-dir "$S0" --source-dir "$S0_PUBLIC" --new-source-dir "$S1_PUBLIC" \
  --tokenizer-path "$DLLM_BASE" --heldout-profile "$FROZEN/profile.json" \
  --output-dir "$S1_SMOKE_NEW" --config "$CODE/research/s1_data_config.json" \
  --seed 42 --max-length 4096 --smoke
```

Smoke limits additions to at most two per quota bucket/split, but intentionally
audits all S0 and all frozen external text and samples the same bounded raw pools.
It does not guarantee every bucket survives exclusions. For the full build:

```bash
python "$CODE/research/scripts/build_shared_yesno_s1.py" \
  --s0-dir "$S0" --source-dir "$S0_PUBLIC" --new-source-dir "$S1_PUBLIC" \
  --tokenizer-path "$DLLM_BASE" --heldout-profile "$FROZEN/profile.json" \
  --output-dir "$S1_FULL_NEW" --config "$CODE/research/s1_data_config.json" \
  --seed 42 --max-length 4096
```

Output must not exist and must be separate from all inputs. Failed builds retain
`report.json` with `status=failed`; use a new output path for retry. Successful
builds report `complete` or `complete_with_shortfalls` and write:

- `canonical/train.jsonl`: retained S0 train followed by additions, trainer-compatible.
- `canonical/dev.jsonl`: exact original S0 selection dev, unchanged bytes.
- `canonical/new_train.jsonl`: additions only.
- `canonical/new_dev.jsonl`: grouped diagnostic additions, rows tagged `split=dev`.
- `source_manifest.json`: original source declarations, revisions, renderer, synthetic lineage.
- `report.json` and `rejections.jsonl`: measured build statistics and candidate-only reasons.

No redundant rendered-token/Parquet export is produced. Successful canonical
exports are reopened for equality checks and original dev bytes are compared.
Review measured counts, shortfalls, and any S0 dev warnings before training;
requested counts are never presented as achieved counts.
