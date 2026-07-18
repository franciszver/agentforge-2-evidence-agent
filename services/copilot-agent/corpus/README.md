# Guideline Corpus (P3.3, non-PHI, public retrieval-test corpus)

This directory is the small, public, non-PHI clinical-guideline corpus that
`app/retrieval.py`'s hybrid (sparse + dense) index is built over, per
`docs/W2_ARCHITECTURE.md` ("Hybrid retrieval-augmented answering") and
`planning/PLAN.md` ("Fully-local tooling choices" / "Corpus").

## What these documents are — and are not

**Every document here is an original, self-authored educational summary
written for this repository**, loosely paraphrasing the *general shape* of
widely known, public-knowledge clinical guidance (the kind of thresholds and
cautions taught in any primary-care curriculum: ADA-style A1c targets,
JNC/ACC-AHA-style blood-pressure categories, common NSAID drug-interaction
cautions, standard statin-monitoring practice, etc.). None of them are
verbatim copies or close paraphrases of any specific copyrighted guideline
publication, and none were scraped from a guideline society's website or
PDF. No document cites a specific version/year of any named guideline body
as its authority — where a family of guidance is referenced (e.g. "ADA-style"
or "JNC/ACC-AHA-style"), that is a description of the general public-domain
shape of the guidance, not a claim of verbatim sourcing.

**These are retrieval-testing fixtures, not clinical advice.** They exist so
`app/retrieval.py`'s hybrid BM25 + dense-embedding index has a small, stable,
topically-relevant corpus to index and evaluate golden queries against. They
are deliberately short, simplified, and not exhaustive — a real deployment
would source current, versioned, authoritative guideline text through a
proper licensing/update process. Nothing in this corpus should be used to
make an actual clinical decision.

**Licensing:** self-authored content, original to this repository. No license
restriction, no attribution requirement, no redistribution concern — unlike
`app/data/drug_interactions_source.csv` (a real third-party dataset's schema
style referenced but not redistributed), these documents contain no
third-party text at all.

## Document format

Each `.md` file starts with a YAML-ish front-matter block:

```
---
id: <doc-id, kebab-case, stable>
title: <human-readable title>
topic: <short topic tag>
uc_mapping: [UC1, UC2, ...]
---
```

followed by `##`-level section headings. Each section becomes one retrieval
chunk with a stable chunk id of the form `<doc_id>#<section-slug>` (the slug
is the heading text, lowercased, with non-alphanumerics collapsed to `-`) —
see `app/retrieval.py`'s `parse_corpus` / `_slugify`. Headings and their
slugs are considered part of the corpus's stable public interface: renaming
a heading changes its chunk id, which changes the golden-query fixtures in
`tests/test_retrieval.py` and the recorded embeddings in
`app/data/retrieval_embeddings.json` (regenerate via
`scripts/build_retrieval_embeddings.py` after any such change).

## Documents

| id | title | topic | UC mapping |
|---|---|---|---|
| `a1c-targets` | A1c Targets for Adults with Diabetes | diabetes-monitoring | UC3 |
| `blood-pressure-categories` | Blood Pressure Categories and Thresholds | hypertension | UC1, UC3 |
| `nsaid-interactions` | NSAID Drug-Interaction Cautions | medication-safety | UC2 |
| `statin-monitoring` | Statin Therapy Monitoring | medication-safety | UC2, UC3 |
| `anticoagulant-interactions` | Anticoagulant and Antiplatelet Interaction Cautions | medication-safety | UC2 |
| `lipid-panel-reference` | Lipid Panel Reference Ranges and Follow-Up | lipid-monitoring | UC3 |
| `hypertension-lifestyle` | Hypertension Lifestyle Management and Follow-Up | hypertension | UC1 |
| `renal-function-monitoring` | Renal Function Monitoring on Nephrotoxic-Risk Medications | medication-safety | UC2, UC3 |
