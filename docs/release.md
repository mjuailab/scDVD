# Refactor and release notes

The original `scDVD/` is backed up in the sibling `scDVD_copy/` before replacing
its source files. Existing large datasets, checkpoints and outputs remain local;
historical source/configs and root logs are retained under `archive/`.
The backup and local validation reports are not part of the source export.

The publication-facing source contains the canonical set4 model, portable config,
checkpoint relocation support, a CPU example, tests, exact paper splits and table
inputs. The manuscript PDF itself is not copied into release bundles.

The source repository is https://github.com/seunghoo123/scDVD.
The project license and public download locations for data/features and
checkpoints are not yet assigned. No license grant, DOI or acceptance venue has
been assumed by this refactor. `CITATION.cff` records the known authors, title,
version and repository URL.
Retain attribution and applicable notices for third-party components when making
a release. GeomLoss is declared as an external dependency rather than silently
embedded in the installed package.

`scripts/export_release.py` uses an explicit allowlist and excludes private local
manifests, historical archives, raw H5AD/checkpoint files and the supplied PDF.
It creates a reviewable archive without uploading, creating a remote repository
or publishing a release.
