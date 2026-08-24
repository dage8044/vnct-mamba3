# VNCT structural reproduction notes

This repository contains an independent, readable PyTorch reproduction of
the architecture described in *Vision Non-Causal Trapezoidal Mamba* (arXiv
2607.03589v2). It is intended as the backbone for later BIQA work, not as a
claim of checkpoint- or benchmark-level reproduction.

## Equation-to-code mapping

- `NCMamba3._coefficients`: equations (11)--(14), including data-dependent
  `DT`, negative decay, and the second-order `alpha`, `beta`, `gamma` terms.
- `NCMamba3._shift_beta`: the one-step-ahead `beta+` endpoint in equation
  (15). The last source receives zero rather than wrapping to the first.
- `NCMamba3.forward`: the two spatial softmax terms in equation (15), the
  chunked MIMO global state in equation (17), readout in equation (18), and
  gated skip/output projection in equations (19)--(20).
- `VNCTBlock`: LPU, absolute `Pos2D`, NC-M3/MHSA residual, and FFN from
  equations (22)--(25).
- `VNCTBackbone`: stages 1--3 use NC-M3, stage 4 uses MHSA, and the returned
  feature pyramid has strides `{4, 8, 16, 32}`.

The production BIQA backbone uses the checkpoint-compatible bridge in
`vssd_small_ncm3.py`. It retains every inherited VSSD projection tensor,
reuses the inherited rank as rank zero, and adds three learned B/C
perturbation ranks. Symmetric `1/sqrt(4)` input/output mixing preserves the
rank-1 magnitude when the ranks coincide. A small token-dependent modulation
is applied to the inherited negative A through an inverse-softplus
parameterization, so its sign is preserved.

## Reproduction boundary

The paper exposes channels, depths, drop-path rates, `d_state=64`, MIMO rank
examples, and chunk-size guidance, but it does not fully specify every model
hyperparameter. The public code URL printed in the paper was not accessible
when this implementation was prepared (2026-08-19). Consequently:

- `ssm_head_dim=64`, `attention_heads=24`, `mlp_ratio=3`, and
  `rope_fraction=0.5` are explicit configurable assumptions based on the
  accompanying Mamba-3 implementation and the VSSD-style hierarchy.
- Equation (10) and its surrounding prose are ambiguous about the projected
  angle/decay streams. This implementation uses the third per-head stream as
  data-dependent decay, matching Mamba-3, and uses fixed grid-based 2D RoPE
  as specified by supplementary section A.4.
- The implementation is a correctness/reference path using standard PyTorch
  einsums. It is chunked for memory, but it is not yet a fused latency kernel.

These choices should be compared with the authors' code if it becomes public.
