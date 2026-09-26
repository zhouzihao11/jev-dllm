# V2 exposure-overlap audit

`audit_overlap.py` is a standard-library-only, offline audit, not a trainer,
converter, release builder, model evaluator, or contamination-free certification.
Python 3.10+ is supported. It never imports the training/evaluation runtime, opens a
checkpoint, or uses model predictions, model scores, or accuracy to select examples.
Gold targets are inspected only to distinguish repeated evaluation annotations
from conflicting targets for identical normalized state-plus-question inputs.
The parent runs it remotely against the actual files and inspects its evidence.
No local execution, data conversion, download, or ML environment setup is needed.

## Inputs and invocation

Use the inherited `full_eval_v2` profile contract validated by
`benchmark/common.py:source_profile` and
`research/scripts/bench_extended.py:load_profile`. The entire ordered profile is read, not a
results directory or a previously filtered pack. Profile paths are relative to
the profile and cannot escape its directory. Every suite's `expected_count` and
globally unique decision IDs are checked. This audit is not a replacement for the
evaluator's complete question/target schema validation.

The current full profile is expected to have 19,826 decisions. Its candidate core
has 17 suites and 6,768 decisions after the **fixed**, pre-scoring exclusion of
`clinc150_oos_seen_source`, `nimble_boolq`, and `forecastbench`. These totals are
reported rather than hard-coded: the parent must verify the intended profile and
these counts. All full-profile rows get a decision, including fixed exclusions.
Mapped sources also present in the exposure inventory are excluded dynamically,
before lexical matching (including the JevBench and jabr source aliases).
Known source-overlap metadata is counted and retained in the report even on those
excluded rows. ForecastBench is excluded for retrospective scope, not as a finding
of fine-tuning leakage.

`--exposures` takes a JSON **file containing a nonempty list**, not inline JSON:

```json
[
  {
    "name": "s1_gradient",
    "role": "gradient_train",
    "path": "/data/actual-s1/train.jsonl",
    "training_manifest": "/runs/actual-s1/run.json",
    "model": "S1",
    "expected_count": 48000
  },
  {
    "name": "s1_selection",
    "role": "selection_dev",
    "path": "/data/actual-s1/dev.jsonl",
    "training_manifest": "/runs/actual-s1/run.json",
    "model": "S1",
    "expected_count": 5000
  }
]
```

Counts and paths in this example are illustrative, not claims about actual runs.
Supply **all** actual S0/S1 training, checkpoint-selection dev, and diagnostic dev
exposures; give separate entries to repeated files used by different models or
roles. Names must be unique. Required fields are `name`, `role`, `path`, `model`,
and `expected_count`; `training_manifest` is optional. Roles are exactly
`gradient_train`, `selection_dev`, or `diagnostic_dev`. Relative exposure and
manifest paths resolve against the exposures JSON directory. Exposure files are
canonical `shared_yesno_data_v1` JSONL, with `case_id`, `view_id`, `group_id`,
`split`, `state`, and **object** `source` containing `name`, `record_id`, and
`original_split`. No conversion is performed.

For each supplied actual run manifest, the audit checks its `config.train_data.path`
or `config.dev_data.path` against the exposure's resolved path and its
`config.train_count` or `config.dev_count` against the observed file count and
`expected_count`. Nonzero `train_limit`/`dev_limit` are rejected: a full file is not
the same as a trainer-selected subset. Both train and dev must be represented for
each model/manifest pair. The inherited trainer writes absolute data paths; the
audit requires those, not guessed relative launch directories. Run ID, code
commit, and existing file metadata are recorded. No audit-only hashing framework
is added. Missing provenance is explicitly reported with an explanation;
the operator must establish that a named model really used the supplied run.
Manifests do not prove historical file immutability or complete gradient history.
Diagnostic files not used by a training run should omit `training_manifest`.
Instead supply `diagnostic_report` pointing to the independently generated
`shared_yesno_supervised.py` report. The audit checks `settings.data` (absolute)
against the exposure path, top-level `count` against `expected_count` and the
observed file count, and requires `settings.limit == 0` and role `diagnostic_dev`.
It records the absolute `settings.model_path`. Supply exposure `model_path` to
check the model association, or use an absolute path as `model`; a symbolic model
alias alone is explicitly marked `declared_alias_only`, not independently verified.
Relative report and declared model paths resolve against the exposures file.
Only these provenance fields are inspected, not report performance metrics or
prediction files. A diagnostic report is not a training manifest and does not
establish checkpoint-selection use. Missing training manifests are listed
separately from missing both forms of provenance.

```bash
python benchmark/audit/audit_overlap.py \
  --profile /data/frozen/full_eval_v2.json \
  --exposures /data/audit/exposures.json \
  --output-dir /data/private-audits/overlap-first
```

Output's parent must already exist. The leaf directory must be new. Prefer a
private directory outside the project because evidence includes full source text.
An interrupted/failed run is not valid: inspect the exit code and require
`audit.json.status == "complete"` plus the final Markdown report. No input or
prior output is overwritten. JSON progress is flushed while loading exposures,
building the index, scanning spans, and writing matches.

## Source identity and text matching

1. **Source namespace.** Explicit aliases and all observed mappings are written
   to `audit.json`. CLINC and BoolQ are matched to the actual canonical sources,
   not stringified dictionaries. MASSIVE English/German share one namespace;
   SummEval dimensions share one. SNLI and MultiNLI are related but **different**
   namespaces. ARC, SGD, DBpedia, and GoEmotions have explicit aliases. Unlisted
   source names, including distinct solver families, remain distinct under
   `unmapped:<name>`; inspect these rather than assuming every solver is one
   dataset. Known Nimble suite names take precedence over upstream URL spelling.
2. **Raw IDs.** Compare original `source.record_id` to evaluation
   `metadata.source.row_id` (or `record_id`) only inside the same namespace.
   Bare decimal IDs additionally require matching original split. IDs with
   prefixes are not shortened; nondecimal IDs are treated as source-local stable
   IDs. Unknown or inconsistently prefixed IDs can therefore miss true overlap;
   text matching remains independent. No local integer ID is compared across
   different datasets. A raw-ID match excludes but is not a label-seen claim.
3. **Whole-state equality.** Unicode NFKC, casefold, and whitespace normalization
   are applied recursively to string values. Dictionary/list structure and order
   remain significant. Whole states with no eligible substantive text are not
   matched. Numeric-only solver facts are outside this lexical audit.
4. **Content-component equality.** Nested string leaves, cleaned concatenated
   state content, labelled sections, and individual lines/paragraphs are compared after
   the same normalization (punctuation is retained). Paths/field labels are
   evidence, not text. Labelled lines strip the field label; instruction, option,
   criteria, rule/policy, and related fields/sections are excluded. A labelled
   ignored section stays ignored until the next label. The entire original state
   is still included in evidence. Exact spans require at least **3 word tokens**.
   Shared content is not automatically a repeated task, gold label, or decision.
5. **Near matching.** NFKC/casefold word tokens (`\w+`, punctuation removed) form
   sets of contiguous **5-word shingles**. A candidate/exposure pair qualifies if
   shingle Jaccard is at least **0.8**, OR shingle intersection divided by the
   smaller shingle-set size is at least **0.9** with at least **12 distinct shared
   word types**. Near spans need at least **12 word tokens**. The first branch
   does not require 12 distinct words; both spans still meet the token minimum.
   A zero-shingle-intersection pair cannot meet either positive threshold.

The CLI exposes `--exact-min-words 3`, `--near-min-words 12`, `--jaccard 0.8`,
`--containment 0.9`, `--shared-words 12`, `--group-min-words 40`, and
`--progress-every 1000`. Threshold changes must be disclosed, not optimized against
model scores. Exact comparisons retain punctuation; lexical near comparisons do
not. Very short utterances below three words are intentionally not text-matched.
Heuristic template exclusion cannot identify every generic instruction or legal
boilerplate; review the private matched spans.

The inverted index is on unique **candidate-core** spans. Exposure views sharing
source namespace, original record ID/split, and identical state are coalesced;
their original case/view/group IDs, roles, models, and exposure names are retained.
Identical exposure spans are scanned once with all record owners attached.
All qualifying candidate/exposure span matches are preserved, not just the first
hit. Postings and comparisons have **no caps**, no common-shingle skipping, and
no probabilistic retrieval. This avoids a 53,000 by 6,768 all-record Cartesian
comparison, though very common long passages can still cost substantial memory,
time, and output space. Full-text evidence is intentionally verbose.

## Decisions, groups, and manual review

Raw identity and exact whole-state equality cause exclusion and cannot be
dismissed. All `exact_content` spans (even full cleaned-content spans) and near
matches default to quarantine pending review, **not confirmed-leak classification**. All three
exposure roles participate conservatively. Evidence preserves the distinction
between gradient exposure, model-selection exposure, and diagnostic exposure.

Within-benchmark duplicate decisions use the full normalized **state plus qdef**,
including ordered instructions/criteria, never state alone. The first occurrence
in profile suite order and then JSONL order is canonical when targets agree;
later occurrences get `duplicate_decision` with `canonical_id`. Target agreement
compares the complete `{gold_idx, soft, gold_score, option_ids}` signature without
text normalization, rounding, averaging, or model-performance selection.
If any pair disagrees, **all** members of that state-plus-qdef family are
quarantined as `duplicate_target_ambiguity`, including the first canonical row,
earlier duplicate-only exclusions, and later members matching either target.
Every pair is recorded in `duplicate_cases.jsonl` with both IDs, target signatures,
and `target_same`. Known-source/fixed exclusions remain exclusions and are outside
duplicate assessment. Different questions or criteria on the same document are
legitimate decisions, not deleted as duplicates. Annotation ambiguity is not
training leakage and does not propagate to other decisions on the same article,
original document group, or upstream family. Duplicate-only exclusions likewise
do not exclude their group or replace an excluded canonical counterpart.

Overlap/quarantine propagates through suite-local `(suite, group_id)` groups and
explicit `(namespace, metadata.upstream_family)` connections, and eligible
shared-content connections. Upstream families require a mapped source namespace;
family links retain the namespace, family, and both IDs in the report. This joins
language variants and source-family reuse across suites without treating shared
family membership alone as exposure. Original group IDs remain suite-local.
Identical full content connects rows within
a source; content of at least `group-min-words` can connect across sources.
Individual document-like whole fields connect only at that minimum length and
when covering at least half the row's eligible words. Isolated line/paragraph
matches, short hypotheses, criterion labels, options, and rule fields do not
create component connections. These restrictions avoid collapsing tasks merely
because a generic instruction or short boilerplate is repeated. Shared states
across suites are logged as links, not automatically deleted. Only actual
exposure findings propagate through those links. Original suite groups are
respected even when broad, so exclusion may be deliberately conservative.
Confirmed exposure exclusions override ambiguity quarantine; ambiguity quarantine
overrides duplicate-only exclusion. Other precedence is `exclude > quarantine > retain`.
Fixed-excluded and dynamically source-excluded rows do not
participate in matching/grouping and cannot contaminate candidate-core groups.

Optional hand review is a JSON file containing exactly this list item schema:

```json
[
  {
    "candidate_id": "candidate/00000017",
    "disposition": "dismiss",
    "reason": "Independent inspection found only generic boilerplate, not shared source content."
  }
]
```

Dispositions are `dismiss`, `confirm_overlap`, or `quarantine`; a nonempty reason
is required. Only `near_content` and `exact_content` evidence can be reviewed
through this interface, never `raw_id` or `exact_whole_state` identity evidence.
Unreviewed content matches stay quarantined. Dismissals are classified
`dismissed_generic_or_distinct_context`; no phrases or candidate IDs are built
into the code. Dismissing one match does not dismiss
other matches or another group member's finding. Confirming a match excludes its
connected group. Reviews and reasons are embedded in all final reports/evidence.
IDs are deterministic enumeration, not hashes: preserve profile/exposure order,
files, and thresholds between runs. Unknown/non-reviewable review IDs fail. The parent
must check evidence identity before applying a review; the tool cannot detect a
stale review that happens to reuse a valid ID on different inputs.

```bash
python benchmark/audit/audit_overlap.py \
  --profile /data/frozen/full_eval_v2.json \
  --exposures /data/audit/exposures.json \
  --review-decisions /data/audit/hand-review.json \
  --output-dir /data/private-audits/overlap-reviewed
```

## Output schema

- `decisions.jsonl`: exactly one row per full-profile decision, in original order:
  `{id, suite, group_id, decision, reasons}`. `decision` is `retain`, `exclude`, or
  `quarantine`. Each reason has `kind`; duplicates add `canonical_id`; propagated
  findings add `component_id`, `direct_hits`, and `evidence_member_ids`. A direct
  hit has `{candidate_id, kind, decision}`. Retained rows can have an empty list.
- `retained_ids.json`: `{profile_id, retained_ids: [id, ...], counts_by_suite}`;
  counts include zero-retention suites. This is the independent pack builder's
  input, not a rewritten benchmark or a `.release` artifact.
- `candidates.jsonl`: `{candidate_id, kind, similarity, classification,
  label_seen, disposition, review, evaluation, exposures}`. Kinds are `raw_id`,
  `exact_whole_state`, `exact_content`, and `near_content`. Similarity is null for
  raw IDs, `{normalized_equal: true}` for whole state, otherwise `{jaccard,
  short_containment, shared_words}`. Classification is `source_record_identity`,
  `substantive_shared_content`, `unresolved_lexical_similarity`, or
  `dismissed_generic_or_distinct_context`; `label_seen`
  is always `not_inferred`. Evaluation entries contain `{id, suite, group_id,
  source, field, matched_span, original_state}`. Exposure entries contain
  `{source, namespace, field, matched_span, original_state, references}`.
  References contain `{exposure, role, model, case_id, view_id, group_id,
  record_id, original_split, split}`. Multiple rows/spans/roles share one evidence
  event when the normalized content is identical; no ownership information is
  intentionally discarded. Raw-ID spans are null; whole-state spans are JSON.
- `duplicate_cases.jsonl`: every candidate-core duplicate pair, canonical ID,
  both target signatures, and `target_same`; no model outputs are used.
- `audit.json`: completion status; profile/exposure paths; raw/core/suite and
  retain/exclude/quarantine counts; exposure raw/unique record/state/span counts;
  source/role/split counts and manifest provenance; aliases and resolved sources;
  source-overlap metadata; thresholds; uncapped comparison and hit counts;
  duplicate count/pair count; target-ambiguity ID list and ID/group counts;
  diagnostic-report provenance; group links/components; reviews; and scope limitations.
- `AUDIT_REPORT.md`: human-readable decision table, provenance/evidence pointers,
  duplicate/ambiguity counts, missing manifests versus missing provenance, and
  explicit interpretation limits.

## Interpretation and validation boundary

This is lexical, **not semantic** matching. Paraphrases, translations, altered
IDs, unavailable datasets, undocumented exposures, and template heuristics limit
recall/precision. Pretraining contamination remains **unknown**. The v2 benchmark
has already been evaluated, so this is **not blind** or prospective. A retained
subset is conditional on the supplied exposure inventory and thresholds; it is
not blanket clean certification. Before use, the parent checks totals, all
source mappings, run manifests, roles, exact hits, near reviews, unusually large
group components, and retained per-suite coverage. This implementation is locally
validated only by static AST/diff inspection; actual audit execution and evidence
inspection belong to the parent's remote run.
Rerun with the same ordered profile/exposures and runtime review JSON in a new
output directory. Pack construction retains only audited retained IDs; the full
original-state and target comparison remains intact and is not relaxed here.
