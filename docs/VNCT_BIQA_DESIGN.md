# VNCT-BIQA architecture design

## 1. Scope and fixed protocol

VNCT-BIQA is a single-backbone NR-IQA model. The production experiment uses
the checkpoint-compatible VSSD-Small backbone and adds stage-wise local
evidence, learned region routing, Region NC-SSD, and a joint quality head. No
distortion label, distortion classifier, second image encoder, or pixel-level
selector annotation is used.

The data and optimization protocol remains the repository's LoDa protocol:

- 224-pixel random crops and the configured LoDa train/test patch counts;
- reference-disjoint splits for synthetic datasets and image-disjoint splits
  for authentic datasets;
- AdamW, cosine scheduling, and the dataset-specific logical batch/T-max
  values already pinned in `configs/experiments/*_learned_loda.yaml`;
- LoDa PLCC loss on the final image-quality prediction only;
- one development split while the architecture is still being established.

ReLIQS motivates the lightweight importance-map decoder and soft spatial
aggregation. It does **not** replace LoDa's training loss, split, crop, or
evaluation protocol. The resulting map is called a *quality-aware importance
map*, not a distortion-probability or uncertainty map.

## 2. Stage-wise data flow

For stage `s`, the backbone produces `R_s: [B,C_s,H_s,W_s]` with VSSD-Small
channels `[96,192,384,768]`. Every stage also produces a full spatial local
feature `L_s`. Stages 1--3 predict an importance map, select a variable number
of ROIs, and refine each ROI independently. Stage 4 keeps its backbone MSA
output and local path but has no selector or Region NC-SSD.

```text
R_s ─────────────────────────────────────────── query ──┐
 │                                                     │
 ├─ multi-range local mixer ─ L_s ─ 7x7 tokens ───────┤
 │                              └─ QA summary token ───┤  stage 1--3
 └─ learned selector ─ ROIs ─ independent NC-SSD ─────┤
                                                       ▼
                                unified cross-attention + fusion ─ Z_s

Z_1, Z_2, Z_3, Z_4 ─ joint-stage quality head ─ quality score
```

## 3. Multi-range local evidence

The local branch is present at stages 1--4 and preserves the full feature map:

```math
U_s=P_s(R_s),
```

```math
V_s=P_v[\operatorname{DWConv}_{3\times3}(U_s);
        \operatorname{DWConv}_{3\times3,d=2}(U_s)],
```

```math
G_s=\sigma(P_g(\operatorname{DWConv}_{3\times3,d=2}
    (\operatorname{DWConv}_{3\times3}(U_s)))),
```

```math
L_s=U_s+\delta_sP_o(\operatorname{FFN}
    (\operatorname{GELU}(V_s)\odot G_s)).
```

Depthwise convolutions diversify the local receptive field without the
quadratic parameter cost of full convolutions at VSSD channel widths. The
pointwise projections still mix channels. The branch performs no GAP, KAN, or
patch-wise pooling before interaction.

## 4. Learned quality-aware importance

Stages 1--3 each decode a spatial map from the main feature `R_s` only:

```math
A_s=\operatorname{softmax}_{HW}\left(P_{1\times1}
(\operatorname{GELU}(\operatorname{DWConv}_{5\times5}
(\operatorname{GELU}(\operatorname{DWConv}_{5\times5}
(\operatorname{LN}(R_s)))))))\right).
```

This ReLIQS-style path provides a normalized distribution `sum(A_s)=1`. It is
learned indirectly from the final MOS objective. It must therefore be
described as identifying locations useful for quality prediction, not as
proving where distortion exists.

The same map creates a continuous, differentiable summary from local evidence:

```math
q_s^L=\sum_{h,w}A_s(h,w)L_s(:,h,w).
```

This is one token inside the evidence bank. It is not an auxiliary score and
has no auxiliary loss. It gives the importance decoder a continuous gradient
path even though ROI indices are selected discretely.

## 5. Budget-relative marginal-coverage routing

The initial setting fixes an odd `5 x 5` feature-space ROI for stages 1--3.
Only complete in-bounds windows are candidates; no padding is used. `3 x 3`
is reserved for a later size ablation.

For candidate window `W_i`, let `C_{k-1}` be the union of previously selected
windows. At iteration `k`, its marginal uncovered gain is

```math
g_{s,k}(i)=\sum_{(h,w)\in W_i\setminus C_{k-1}}A_s(h,w).
```

The maximum-gain candidate is selected and its footprint is added to the
covered set. Overlapping or adjacent candidates are allowed only when they add
new importance mass; already covered cells contribute zero. This replaces
zero-IoU NMS and avoids declaring nearby quality evidence invalid.

The procedure obtains at most `K_max=4` gains before Region NC-SSD is run. The
gains are normalized within that four-candidate budget:

```math
\bar g_{s,k}=\frac{g_{s,k}}{\sum_{j=1}^{K_{max}}g_{s,j}},
```

and the active count is the smallest count reaching `tau=0.8`:

```math
K_s=\min\left\{k:\sum_{j=1}^{k}\bar g_{s,j}\ge0.8\right\}.
```

Thus each image and stage uses one to four ROIs. The 0.8 value is a
budget-relative threshold over four proposed gains, not 80% of the full
image's probability mass. Hard indices and the active count are routing
decisions; no claim of differentiability through argmax is made.

## 6. Independent Region NC-SSD

Each active ROI is cropped from `R_s`, not from `L_s` or `[R_s;L_s]`. ROIs are
processed independently by the shared stage-specific Region NC-SSD:

```math
\widetilde X_s^k=X_s^k+\gamma_s
\operatorname{NCSSD}_s(\operatorname{LN}(X_s^k)).
```

The odd ROI has a unique center. Only its refined center token is retained:

```math
e_s^k=\widetilde X_s^k[:,\lfloor r/2\rfloor,\lfloor r/2\rfloor].
```

There is no joint sequence across ROIs, `2 x 2` ROI pooling, or scalar
importance multiplication of a regional value. Invalid padded slots are
masked in attention. Every regional token receives its normalized ROI-center
coordinate.

## 7. Unified evidence-bank interaction

The local map is adaptively pooled to `7 x 7`, yielding 49 local tokens. At
stages 1--3 the evidence bank is

```math
B_s=[\underbrace{\operatorname{AAP}_{7\times7}(L_s)}_{49\ local};
     \underbrace{q_s^L}_{1\ summary};
     \underbrace{e_s^1,\ldots,e_s^{K_s}}_{1\text{--}4\ regional}].
```

Stage 4 uses only the 49 local tokens. Local and regional tokens receive
continuous normalized 2-D sinusoidal positions. The summary has no spatial
coordinate and instead uses a learned global/source embedding. Learned
source-type embeddings distinguish local, summary, and regional evidence.

The original backbone feature remains the query:

```math
M_s=\operatorname{CrossAttention}(Q=R_s,K=B_s,V=B_s),
```

and interaction generates a new representation by feature fusion:

```math
Z_s=P_{out}(\operatorname{GELU}(P_{mix}([R_s;M_s]))).
```

This replaces residual correction
`R + alpha*C(R,L) + beta*C(R,E)`. One attention normalizes competition among
all evidence types, after which fusion learns the main/evidence composition.
No subtraction feature is used. Complexity is `O(N_s(50+K_s))` at stages
1--3 and `O(49N_4)` at stage 4.

## 8. Quality head and experiment claims

The existing joint-stage quality head remains fixed for the first experiment.
Each `Z_s` is aligned to `7 x 7`, projected to a common dimension, and the four
stage token sets interact in parallel before spatial and stage-weighted score
pooling. No top-down stage cascade is introduced.

Initial claims are limited to quality-aware routing, variable computation
within a four-ROI budget, region-specific NC-SSD evidence, unified local/region
interaction, and evaluation under unchanged LoDa settings. Whether the map
localizes visible distortion, whether `5 x 5` is optimal, and whether dynamic
K improves performance are empirical questions requiring visualization,
K-distribution, ablation, parameter/FLOP, and SRCC/PLCC evidence.

## 9. Primary references

- LoDa, *Boosting Image Quality Assessment through Efficient Transformer
  Adaptation with Local Feature Enhancement*, CVPR 2024:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.html>
- ReLIQS, *Learning Where to Look and How to Judge: Resolution-agnostic Image
  Quality Assessment*, CVPR 2026:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Gedik_Learning_Where_to_Look_and_How_to_Judge_Resolution-agnostic_Image_CVPR_2026_paper.html>

The decoder and quality-aware soft aggregation are adaptations of the ReLIQS
mechanism. Stage-wise routing, marginal uncovered coverage, dynamic K, Region
NC-SSD, center-token extraction, and the unified evidence bank are the proposed
VNCT-BIQA design and must not be attributed to ReLIQS.
