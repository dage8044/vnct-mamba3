# VNCT-BIQA implementation plan

## 1. Canonical target

The learned-routing architecture is defined by
`configs/models/vnct_biqa_vssd_small_learned.yaml`. The LoDa experiment YAML
files continue to define dataset paths, splits, crops, optimizer, scheduler,
logical batches, PLCC loss, and evaluation. The architecture file controls
only model behavior.

Implementation order:

1. Freeze the tensor contract in `docs/VNCT_BIQA_DESIGN.md`.
2. Replace fixed-K NMS and the auxiliary predictor.
3. Replace joint ROI processing and four-token pooling.
4. Replace dual residual attention with unified evidence-bank interaction.
5. Wire configuration, diagnostics, visualization, and tests.
6. Run focused tests and a debug forward/backward.

## 2. Module contracts

### LearnedImportanceSelector

Input: `R_s [B,C,H,W]`. Output `ImportanceSelection`:

- `score_map [B,1,H,W]`, spatial-softmax normalized;
- `boxes [B,4,4]` and normalized `centers [B,4,2]`;
- `indices [B,4]`, `marginal_gains [B,4]`, and `gain_shares [B,4]`;
- `valid_mask [B,4]` and `num_selected [B]` from threshold 0.8.

The selector uses valid stride-one `5 x 5` candidates, greedy marginal
uncovered mass, and no NMS. It contains no quality regressor or auxiliary
score.

### SelectedRegionRefiner

Input: `R_s`, boxes, and `valid_mask`. Each of four padded slots is aligned to
`5 x 5` and independently passed through the shared stage Region NC-SSD.
Output is `tokens [B,4,C]`, `coordinates [B,4,2]`, `valid_mask [B,4]`, plus ROI
features for inspection. It never mixes ROIs or multiplies tokens by selector
scores.

### UnifiedEvidenceInteraction

Input: `R_s`, `L_s`, optional importance map, and optional regional
tokens/coordinates/mask. Stages 1--3 build 49 local tokens, one quality-aware
local summary, and one to four regional tokens. Stage 4 builds 49 local tokens.
One masked cross-attention reads the bank. Two-layer concat fusion
`[R_s; attended] -> Z_s` replaces residual alpha/beta updates.

## 3. Machine-readable defaults

```yaml
selector:
  name: learned_importance
  stages: [0, 1, 2]
  region_sizes: [5, 5, 5]
  max_regions: 4
  coverage_threshold: 0.8
  candidate_stride: 1
  decoder_kernel_size: 5
  auxiliary_loss_weight: 0.0

refinement:
  stages: [0, 1, 2]
  roi_sizes: [5, 5, 5]
  independent_regions: true
  output: center_token

interaction:
  name: unified_evidence_bank
  local_grid_size: 7
  local_tokens: 49
  quality_summary_tokens: 1
  max_regional_tokens: 4
  source_type_embedding: true
  fusion: concat_linear_gelu_linear
```

## 4. Training integration

- Final LoDa PLCC is the only objective.
- The map learns through its soft local-summary token; hard routing remains
  non-differentiable.
- `new_modules_only` remains the default policy.
- Alpha/beta gate calibration is disabled because those gates no longer exist.
- CSV diagnostics record selector entropy, peak ratio, selected-K statistics,
  gain shares, selector gradients, and Region NC-SSD residual scales.
- Terminal output remains limited to timestamp, epoch time, loss, quality
  metrics, and best-score notices.

## 5. Required verification

Tests cover spatial-softmax normalization, valid border-contained windows,
greedy marginal gains, cumulative-threshold K, independent ROI center tokens,
attention masking, evidence counts, finite gradients, and debug model
forward/backward. Integration additionally checks checkpoint compatibility,
training-policy groups, serialization, and production shape/FLOP execution.

## 6. Deferred ablations

Deferred choices are `3 x 3` ROI, fixed K, other coverage thresholds,
auxiliary selector losses, joint ROI processing, multiple ROI tokens, scalar
regional weighting, KAN, and a different final quality head.
