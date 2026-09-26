# Portable core preparation

From the release root, one command acquires the pinned sources, rebuilds the
original v2 records, replays the frozen public membership and invokes the existing
`benchmark/prepare.py` packager:

```bash
python scripts/prepare_core.py --output-dir /path/to/new/core-v1
```

Python 3.10+ is supported. Download-only acquisition requires only the standard
library; if `huggingface_hub` is installed, its existing download path is used.
Conversion requires `pyarrow` and, on Python 3.10, `tomli` (the inherited adapter
also supports pip's vendored fallback). No PyTorch, datasets library, model, tokenizer,
accelerator, private training files or private audit inventory is needed for data
preparation. Model evaluation has separate runtime requirements.

The wrapper creates **17 suites, 6768 candidate records, 6744 retained decisions**.
It does not publish anything. The output remains `core_v1_candidate`, not a new
blind benchmark or a claim of contamination-free data. Pass the generated directory
to `benchmark/evaluate.py` using that evaluator's bundle/model CLI.

## Local-first acquisition, remote-only conversion

The release verification workflow deliberately separates acquisition from execution:

```bash
# Authorized acquisition host only: no converter imports or execution.
python scripts/prepare_core.py --download-only --cache-dir /path/to/external/cache

# CPU conversion host after the source tree has been transferred by its operator.
python scripts/prepare_core.py --offline --raw-root /path/to/transferred/raw \
  --output-dir /path/to/new/core-v1
```

No downloader should be run on the experiment server for this verification. The
current preparation owner performs only local source/AST review; the parent owns
remote conversion, evaluator validation and comparison to the research freeze.
`--download-only --offline --raw-root ...` can check required-file availability
without converting anything. It does not establish content equivalence.

The existing uploaded tree with `core/`, `nimble/` and `document_forecast/` is
accepted directly by `--raw-root`; no dataset copying/rearrangement is necessary.
The earlier acquisition layout with `jevbench/` and `classifier-benchmark/` directly
under the raw root is also accepted. If both layouts are present, the direct
layout is selected when `jevbench/` exists. Avoid mixing versions/layouts.

`--raw-root` is an explicit assertion that these are the documented frozen source
files: it is read-only and never downloads, even without `--offline`. Missing or
empty required files fail together with an actionable list. Extra BoolQ, CLINC or
ForecastBench raw files already in that tree are ignored. `--offline` forbids the
wrapper's acquisition calls; the selected converter paths themselves contain no
network requests, and subprocesses also receive the usual HF offline settings.
This is not an OS-level sandbox for arbitrary substituted upstream code.

## Cache and download inventory

The default cache is `$XDG_CACHE_HOME/shared-yesno-dllm` or
`~/.cache/shared-yesno-dllm`, outside the checkout. `--cache-dir` overrides it.
Prepared-source files live beneath `CACHE/core-v1/raw`; when `huggingface_hub` is
installed, HF's own download cache is `CACHE/hf`. The standard-library fallback
writes directly to the raw source destination. Keep both raw sources and generated
bundles out of Git. A bundle contains third-party evaluation content; successful
preparation is not permission to redistribute it.

```text
CACHE/core-v1/raw/
  core/jevbench/datasets/public/{original,easy,hard}.jsonl
  core/classifier-benchmark/cases/v2.toml
  nimble/upstream/nimble/...
  nimble/upstream/docs/assets/public-benchmarks/subsets/*-manifest.json
  nimble/vitaminc.zip
  nimble/amazon-massive-dataset-1.1.tar.gz
  nimble/multinli_1.0.zip
  nimble/squad2/squad_v2/validation-00000-of-00001.parquet
  nimble/paws/labeled_final/test-00000-of-00001.parquet
  nimble/civil_comments/data/test-00000-of-00001.parquet
  nimble/aegis2/test.json
  nimble/helpsteer2/validation.jsonl.gz
  nimble/summeval/data/test-00000-of-00001-35901af5f6649399.parquet
  nimble/pubmedqa/pqa_labeled/train-00000-of-00001.parquet
  document_forecast/contract-nli.zip
```

The exact sources, split files, revisions, inherited license labels and acquisition
URLs are controlled by `benchmark/manifests/core_v1.json`:

- JevBench: `fstandhartinger/jevbench` at `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`.
- jabr: `jabr/classifier-benchmark` at `e5057043852d57c006f0171fc0cddd8ddaf280ad`.
- Nimble converters and official ID manifests: `bespokelabsai/nimble` at
  `62076b4f2d365b5879dafcf7f6dd072a1fe76df7`.
- ContractNLI: official `resources/contract-nli.zip` at
  `eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`; only its test JSON is converted.
- VitaminC: commit-addressed source-site ZIP; MASSIVE 1.1 and MultiNLI 1.0:
  version-named original archives. Historical archive SHA-256 identities are
  recorded, not a new custom integrity-checking framework. Reused raw trees must
  match those snapshots; presence checks are not cryptographic verification.
- Seven HF files only: SQuAD2 validation, PAWS labeled-final test, Civil Comments
  test, Aegis2 test, HelpSteer2 validation, SummEval test, PubMedQA labeled train.
  Every HF call specifies a full revision and exact filename; no latest revisions,
  streaming dataset service, dynamic schema conversion or snapshot-wide download.

There are ten Nimble raw files for twelve selected subsets: MASSIVE shares an
archive across two locales and SummEval shares a file across two criteria. BoolQ
is neither downloaded nor converted. CLINC and ForecastBench are not acquired.
Source archives can contain other splits/locales, and GitHub tarballs can contain
unneeded repository files in transit; only selected code/docs/data paths are
extracted. No downloaded third-party source code or raw dataset is Git-tracked.
Pinned Nimble code is executed from the external raw cache, with upstream notices
retained there; its model/scoring runtime is not instantiated.

Downloads use bounded retries, streaming, temporary files and atomic renames.
GitHub tar extraction rejects path traversal, links and special selected members.
Interrupted file downloads do not become complete cache entries. An incomplete
preexisting GitHub directory fails explicitly rather than merging partial trees;
move it aside and reacquire. There are no automatic source substitutions or
silent partial suites. HF access failures explain HTTP 401/403 without printing
credentials; obtain upstream access if required and set `HF_TOKEN` in the
environment (or use an already authorized HF login with `huggingface_hub`). When
that library cannot be imported, acquisition downloads only the exact manifest
revision/file through a quoted `https://huggingface.co/datasets/.../resolve/...`
URL, using the same retries, per-source progress, nonempty/content-length checks
and atomic writes as other URL downloads. Optional `HF_TOKEN` authorization is
sent only to the original HF origin; redirects changing host/port or scheme strip
the header. Errors omit tokens and URL query strings. No installation, revision
substitution, model download or conversion is performed by this fallback.

## Frozen semantics and public audit

The three migrated adapters preserve their inherited dataset transformations and
metadata. Their only execution deltas are `--include-source` for core/documents
and `--exclude-subset` for Nimble; direct adapter defaults still include their
historical full selection. The wrapper requests JevBench+jabr, non-BoolQ Nimble,
and ContractNLI explicitly. Official Nimble ID manifests define selection, with
no tokenizer refiltering, resampling, inferred ground-truth-based choices or
invented rows. Alias labels are not used for membership selection.

Contract text is never truncated, evidence spans never enter inputs, choice key
order is unchanged, and all original IDs/group IDs and target semantics survive.
SummEval retains the official rounded-half-up hard index and the unrounded
zero-based expert mean as `gold_score`. Human-vote soft targets remain distinct
from exact-mechanism JevBench probabilities. Input construction does not consult
the frozen audit decisions' exclusion reasons or gold values.

The public control includes `benchmark/manifests/core_v1_retained_ids.json`, an
ordered list of the 6744 public IDs extracted from the frozen research selection.
Preparation requires exact equality with that list, not just matching totals.
The remaining compact control uses pinned source membership and explicit exclusions:
four MASSIVE exposure/group rows (IDs 258 and 86 in both locales), seventeen
ContractNLI duplicate hypotheses for contract 97, one SummEval duplicate, and two
SummEval target-ambiguity quarantines. Membership replay validates every candidate
ID, exclusion group, canonical duplicate ID, suite count and final total against
that control. It does not rerun the historical lexical audit against private data.

The generated `retained_ids.json`, `decisions.jsonl`, `audit.json` and
`AUDIT_REPORT.md` are scoped to the **6768-row candidate**, not the historical
19826-row, 20-suite parent. They retain the parent provenance ID
`full_eval_v2_20260925` for historical rescore compatibility while stating this
narrower scope. Status `complete` means frozen membership replay finished, not
that a new exposure audit ran. The public audit includes only IDs, groups, reason
types, counts and historical method parameters. It does not copy the private
audit, host paths, training snippets, exposure inventory or logs.

Retained floors are exact: ContractNLI 2074, MASSIVE 348 per locale, SummEval
relevance 237. All other suite counts are in the control manifest. Policy reasons
are CLINC/BoolQ **seen-source** and ForecastBench **retrospective forecasting**;
ForecastBench is not labeled as known source overlap.

`--output-dir` must be new. Work is staged in a temporary sibling directory, which
is removed on failure, and the finished package is moved into place only after
packaging/count checks succeed. Outputs are `manifest.json`, `inputs.jsonl`,
`targets.jsonl`, `provenance.jsonl`, the four public audit files, and the inherited
`nimble_upstream_checksums.json` report. Official upstream checksum comparisons
are reported rather than used to silently replace data or change membership.

Before accepting release preparation, the parent must remotely rebuild and compare
the ordered input and target JSON values with the frozen 6744-row research
package, check all 17 counts and retained IDs, inspect the checksum report, and run
the evaluator smoke path. Static AST/diff review alone does not establish those
results. Public audit/provenance differences are intentional; input/target
differences are not. Licensing/source-notice review remains separate.
