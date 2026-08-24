# BIQA project structure

The repository separates reproducible experiment settings from reusable code:

```text
configs/
  datasets/       dataset paths and content-separated split policies
  experiments/    backbone, head, optimization, and evaluation settings
  models/         canonical architecture settings shared across datasets
vnct/
  data/           metadata readers and dataset adapters
  engine/         training/evaluation loops (after head selection)
  losses/         differentiable BIQA objectives
  metrics/        SRCC, PLCC, KRCC, and RMSE
  models/
    backbones/     VSSD-Small and VNCT feature extractors
    heads/         parallel four-stage BIQA quality heads
    interactions/  spatial-reduced local/refinement cross-attention
    refinement/    ROI-aligned NC-SSD selected-region processing
    selectors/     fixed or learned spatial evidence for refinement
  utils/          strict checkpoint conversion/loading
tools/             Python entry points for train/test/inspection
tests/vnct/        architecture, checkpoint, and numerical tests
third_party/       pinned research code used for provenance
checkpoints/       local pretrained weights (Git-ignored)
data/              local datasets and metadata (Git-ignored)
outputs/           logs, predictions, and weights (Git-ignored)
```

The controlled BIQA model is composed around the camera-ready VSSD-Small
backbone and official ImageNet checkpoint. The identity profile starts every
added residual at zero for exact feature equivalence. The performance profile
uses rank-4 MIMO, data-dependent A, `0.01` NC-M3/local/refinement residuals,
and interaction gates calibrated to a 2% per-source update norm.

Every dataset split must be reference-content separated. Store generated split
manifests or seeds with the experiment results, and report the median SRCC/PLCC
over repeated splits when following the protocol of the selected BIQA dataset.
