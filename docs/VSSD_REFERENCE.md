# VSSD reference integration

This repository keeps the upstream VSSD implementation as a pinned Git
submodule at `third_party/VSSD`. The upstream repository does not currently
publish a repository-level license, so its source is retained for provenance
and comparison rather than copied into the `vnct` package.

The code under `vnct/` is an independent, BIQA-oriented implementation of the
published NC-SSD structure:

- `vnct/models/layers/`: scan-free token mixers and 2D position encoding
- `vnct/models/backbones/`: multi-scale vision feature extractors
- `vnct/models/heads/`: downstream BIQA heads
- `configs/`: experiment/model structure declarations
- `tests/vnct/`: shape, numerical, and gradient checks

The VSSD backbone deliberately returns feature maps at strides 4, 8, 16 and
32 and does not include an ImageNet classification head.

Clone this project with its reference implementation:

```bash
git clone --recurse-submodules https://github.com/dage8044/vnct-mamba3.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```
