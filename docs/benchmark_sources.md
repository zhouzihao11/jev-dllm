# Benchmark source notices

Reviewed 2026-09-26 for the planned audited Core release (17 suites, 6,744
decisions). This is a source/license review, not a recount or certification of
that membership. Scope: JevBench public, Jabr v2, Nimble12 (without BoolQ), and
ContractNLI. Older data notices remain in [THIRD_PARTY.md](../THIRD_PARTY.md),
[SOURCES.md](../datasets/SOURCES.md), and [LICENSES.md](../datasets/LICENSES.md).
MultiNLI below is **not SNLI**; the older SNLI notice does not cover it.

All licenses below are **upstream-declared, not a blanket legal guarantee**.
Neither this release's code license nor an upstream code repository's MIT
license automatically covers incorporated corpora, third-party text, privacy
rights, or model outputs. Public accessibility is not permission for every use.
New raw third-party data and reconstructed text-bearing subsets are not intended
to be committed; public membership IDs and suite counts are not a relicensing
of the corresponding records.

## Sources and boundaries

| Source | Upstream declaration and inspected source URL | Access / rights notes |
| --- | --- | --- |
| JevBench public (original/easy/hard) | MIT for authored public decisions and harness; [repository](https://github.com/fstandhartinger/jevbench/tree/1bcc55eb6c8cffde2306b3db03ede39b61c6152a), `LICENSE`, `README.md`, `RESULTS.md`, `datasets/HARD-TIER.md`. | Public only, no sealed items. README explicitly covers the original 72; HARD-TIER explicitly identifies public hard as MIT, and RESULTS calls published decisions MIT. Imported third-party decisions and other projects retain their own terms. Hard-tier labels are LLM-authored/reviewed, not human annotations. |
| Jabr v2 | CC0-1.0; [LICENSE](https://github.com/jabr/classifier-benchmark/blob/e5057043852d57c006f0171fc0cddd8ddaf280ad/LICENSE) explicitly includes code, benchmark suite, test cases, and results. | Public `cases/v2.toml`; synthetic upstream benchmark, not human-labeled ground truth. CC0 is not a warranty concerning third-party rights. |
| VitaminC (Nimble12) | CC BY-SA 3.0 in [author's dataset card](https://huggingface.co/datasets/tals/vitaminc); [official repository / archive link](https://github.com/TalSchuster/VitaminC). | Wikipedia revisions plus synthetic revisions. Preserve attribution, source provenance and applicable share-alike obligations for adapted text; a code license does not replace Wikipedia rights. Archive used is `vitaminc.zip`, not a substitute corpus. |
| MASSIVE 1.1 en-US / de-DE (two suites) | CC BY 4.0 for dataset; [Amazon NOTICE](https://github.com/alexa/massive/blob/main/NOTICE.md), [official source and download links](https://github.com/alexa/massive). | Public 1.1 archive. Copyright Amazon.com, Inc. or affiliates; SLURP seed text is also declared CC BY 4.0. Retain Amazon/SLURP attribution and identify adaptations; upstream asks citation of both papers. |
| SQuAD2 | CC BY-SA 4.0; [dataset card](https://huggingface.co/datasets/rajpurkar/squad_v2). | Wikipedia passages and crowdworker questions; retain source attribution and applicable share-alike obligations. Answerability conversion does not remove passage rights. |
| PAWS (Wiki labeled_final) | Custom permissive declaration, HF metadata `other`; [dataset card](https://huggingface.co/datasets/google-research-datasets/paws). | Card says freely usable for any purpose, appreciates Google LLC acknowledgment, and disclaims warranties/liability. Wikipedia-derived text retains underlying rights. This is PAWS-Wiki, **not PAWS-QQP**: card expressly withholds QQP data because of its license and requires separate reconstruction. Do not label all PAWS Apache/MIT. |
| MultiNLI 1.0 dev_matched | Mixed source terms, not one blanket license; [NYU source / original ZIP](https://cims.nyu.edu/~sbowman/multinli/), [NYU dataset card licensing section](https://huggingface.co/datasets/nyu-mll/multi_nli). | Majority under OANC terms; fiction includes CC BY-SA 3.0 (`Seven Swords`), CC BY 3.0 (`Living History`, `Password Incorrect`), and works public-domain in the US but potentially different elsewhere. Preserve source-specific notices; no substitution of SNLI's license. Uses original archive for annotator votes. |
| Civil Comments | CC0-1.0; [Google dataset card](https://huggingface.co/datasets/google/civil_comments) explicitly includes underlying comment text. | Public HF source is used, not a claim of accepting Kaggle competition terms. Toxicity/privacy risks remain despite CC0; do not infer absence of personal information. |
| Aegis2 | **CC BY 4.0 at the exact selected revision**, [NVIDIA card](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/blob/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c/README.md). | Verified rather than inferred from another Aegis version. See revision/access finding below. Human prompt labels only; no reconstruction of redacted Suicide Detection records. Card specifies non-identification, legal compliance and safety/research-oriented use, and warns against dialogue-agent training. Incorporated source rights still matter. |
| HelpSteer2 | CC BY 4.0; [NVIDIA dataset card](https://huggingface.co/datasets/nvidia/HelpSteer2). | Card describes mostly ShareGPT-derived prompts, in-house model responses and Scale AI annotations. Attribute NVIDIA/creators and identify adaptations; dataset declaration is not blanket clearance of every contributed prompt. No model download/license is needed merely to obtain the data. |
| SummEval relevance / consistency (two suites) | MIT metadata in [pinned MTEB export card](https://huggingface.co/datasets/mteb/summeval/blob/bfc121155064afa2d81b5505682ffc0d96f4334c/README.md); [original SummEval source](https://github.com/Yale-LILY/SummEval). | **Underlying CNN/DailyMail article rights remain separate.** Original authors explicitly distribute summaries/annotations without source articles and instruct users to obtain/pair those separately. MIT metadata does not establish unrestricted rights to news text included in an export. Complete text redistribution clearance remains pending. Cite SummEval/MTEB and applicable original summary-model papers as requested upstream. |
| PubMedQA | MIT metadata in [pinned dataset card](https://huggingface.co/datasets/qiaojin/PubMedQA/blob/9001f2853fb87cab8d220904e0de81ac6973b318/README.md); card's detailed licensing section is unfilled. | **PubMed abstracts may be copyrighted by publishers/authors**; see [NCBI policy](https://www.ncbi.nlm.nih.gov/home/about/policies/). MIT metadata is not permission for every abstract. Article-level redistribution rights remain pending; retain PubMed IDs/provenance. |
| ContractNLI | CC BY 4.0 explicitly for the **dataset**, not just repository files; [pinned official site source, License and Download sections](https://github.com/stanfordnlp/contract-nli/blob/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/index.md), [official site](https://stanfordnlp.github.io/contract-nli/). | Official `resources/contract-nli.zip`. Site says accessing/downloading/using means agreement to its Terms and Conditions; read them before download. Attribute Yuta Koreeda and Christopher D. Manning (2021), retain contract source URLs and mark adaptations. Underlying contracts/linked sites are not blanket-cleared by this review. |

## Revision, access and downloader findings

- **Aegis2:** the exact revision's card declares CC BY 4.0 in both metadata and
  prose. The [HF revision API](https://huggingface.co/api/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/revision/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c)
  reports `private: false`, `gated: false`; its file listing has no separate
  license file. A request for `LICENSE` returned 404. No custom NVIDIA dataset
  license was found at this pin: do not substitute an assumed custom license or
  another version's terms. This does not erase the card's use/privacy notices.
- **ContractNLI:** explicit dataset licensing and download terms were inspected
  at the pinned commit. Its dataset license is therefore not marked unknown
  merely because the ZIP lives in a GitHub repository. This review did not
  independently inspect the ZIP's embedded notices; preserve any provided there.
- Other linked source documentation/cards were publicly readable in this review;
  this is not a fresh anonymous-download or gating test for every data file.
  No raw data was downloaded. Parent acquisition notes supply source pins, not
  legal clearance. A successful earlier download does not certify future access.
- All downloads must be user-initiated under the user's own compliance with
  upstream licenses, access terms and any authentication/acceptance requirements.
  Expose source/terms links before download, respect refusals/gates, and do not
  bypass them with mirrors. Publishing a downloader is **not a license solution**.
- No inspected term establishes a categorical prohibition on publishing a simple
  source-linking downloader. This is not unconditional approval of all use or
  redistribution. ContractNLI requires user agreement; SummEval/news and
  PubMedQA/abstract redistribution clearance remains pending. Unknown or
  unavailable terms must remain visibly pending, not certified redistributable.
  These scoped qualifications do not halt unrelated code preparation.

## External code and derived artifacts

JevBench runner code is intended to be a pinned external download, not vendored.
If copied or substantially adapted into a distributed release, retain its full
MIT permission/disclaimer and copyright notice: `Copyright (c) 2026 Florian
Standhartinger and contributors`. A link alone is not a substitute for required
notices in distributed copies. Preserve licenses/notices in external checkouts.

For any later text-bearing redistribution, reassess the exact records and source
terms, retain applicable attribution/license notices, identify selection and
format/label changes, and satisfy applicable share-alike obligations. Neither
membership manifests nor rounded/converted labels extinguish underlying text
rights. No restricted cards, raw records or license-bearing archives are copied
into this document, and this review does not authorize publication of them.
