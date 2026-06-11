# DeepCASE — AnomaLog-maintained fork

This repository is a maintained fork of the original DeepCASE implementation by
van Ede et al. It is used within the
[AnomaLog](https://github.com/harens/anomalog) project for reproducibility,
paper-to-code auditing, and controlled experimentation.

This is **not** the official DeepCASE repository. The goal of this fork is to
preserve the released implementation as far as possible, apply confirmed
maintenance fixes, and document divergences between the paper specification and
the released code.

For the full audit, see:

* [`docs/deepcase_paper_to_code_audit.md`](docs/deepcase-audit.md)

## Relationship to upstream

The upstream DeepCASE repository contains the code released by the authors of
the IEEE S&P DeepCASE paper:

> DeepCASE: Semi-Supervised Contextual Analysis of Security Events.

The upstream `main` branch provides DeepCASE as an out-of-the-box tool. The
original paper experiments are associated with the upstream `sp` branch.

This fork keeps the upstream behaviour as the maintained baseline wherever
possible. Changes that intentionally alter model behaviour are kept separate
from the baseline so that their effects can be evaluated independently.

## What changed in this fork?

This fork includes maintenance fixes for confirmed implementation issues,
including:

* incorrect loss-progress reporting;
* label-smoothing configurability where the exposed parameter was previously
  hardcoded;
* `Interpreter.fit_predict()` not respecting caller-provided prediction
  parameters;
* more reliable `ContextBuilder.load()` behaviour for non-default architecture
  settings.

Paper-facing behavioural alternatives, such as frequency-based label smoothing,
padding-mask semantics, or paper-aligned decoder dimensions, are treated as
experimental variants rather than silent baseline changes.

## Audit status

The audit found that the released implementation should be understood as a
specific implementation of the DeepCASE design, not as a complete one-to-one
encoding of every detail in the paper.

In this fork:

* confirmed bugs are fixed in the maintained baseline;
* ambiguous behaviours are preserved and documented;
* paper divergences are documented explicitly;
* paper-faithful alternatives are isolated as experimental variants.

This makes the fork suitable for controlled AnomaLog experiments where
behavioural provenance matters.

## Original DeepCASE summary

DeepCASE is a semi-supervised approach for contextual analysis of security
events. It learns correlations in sequences of security events, clusters
similar contexts, and presents these clusters to security operators for policy
assignment. Operators can then choose whether to ignore, inspect, or otherwise
act on related groups of events.

The central goal of DeepCASE is to reduce the number of manual inspections that
security operators must perform when triaging large volumes of security events.

## Documentation

Original DeepCASE documentation is available at:

* https://deepcase.readthedocs.io/en/latest/

Fork-specific audit notes are available in this repository under:

* [`docs/`](docs/)

## Datasets

The original DeepCASE evaluation used two datasets:

1. Lastline dataset.
2. HDFS dataset.

The Lastline dataset was obtained under NDA and is not publicly available.

HDFS reproduction work in AnomaLog depends on public and preprocessed HDFS
artefacts. Results should therefore be interpreted through the dataset,
preprocessing, and metric contracts documented in AnomaLog and the audit notes.

## Citation

Please cite the original DeepCASE paper when using DeepCASE in academic work.

```bibtex
@inproceedings{vanede2020deepcase,
  title={{DeepCASE: Semi-Supervised Contextual Analysis of Security Events}},
  author={van Ede, Thijs and Aghakhani, Hojjat and Spahn, Noah and Bortolameotti, Riccardo and Cova, Marco and Continella, Andrea and van Steen, Maarten and Peter, Andreas and Kruegel, Christopher and Vigna, Giovanni},
  booktitle={Proceedings of the IEEE Symposium on Security and Privacy (S&P)},
  year={2022},
  organization={IEEE}
}
```

## Licence

This fork follows the licence terms of the upstream DeepCASE repository.
