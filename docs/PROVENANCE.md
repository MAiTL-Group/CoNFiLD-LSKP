# Source and distribution provenance

This repository extends CoNFiLD with the LSKP prior, diagnostics and portable
training/evaluation entry points. Inherited components and local changes are
listed in [third-party notices](../third_party_licenses/README.md).

## Upstream sources

- CoNFiLD code: https://github.com/jx-wang-s-group/CoNFiLD
- Article: https://doi.org/10.1038/s41467-024-54712-1
- Original data: https://doi.org/10.5281/zenodo.14037782
- Original model archive: https://doi.org/10.5281/zenodo.14058363

The data and model records credit Pan Du, Meet Hemant Parikh, Xiantao Fan,
Xin-Yang Liu and Jian-Xun Wang (2024) and specify
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution,
source, license and modification information when redistributing these assets.
The self-training workflow needs the data archive, not the upstream models.

Our code additions use [Apache-2.0](../LICENSE); inherited Apache/MIT notices
remain applicable. Code licensing does not override external asset terms.
This repository distributes code only; preprocessing is described in the
[README](../README.md) and per-case configurations.
