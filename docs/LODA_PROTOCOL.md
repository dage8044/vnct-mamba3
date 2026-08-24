# LoDa-derived protocol and current four-dataset scope

The experiment config follows the released LoDa code at commit
`82304c20c34c1b5bd45f27bd7ab6e9104a285152`:

- `config/job/train_loda_koniq10k.yaml`: seed 3407, 10 epochs, train/test
  patches 3/15, batches 128/512, and 10 workers.
- `config/data/koniq10k.yaml` and `scripts/process_koniq10k.py`: 10,073
  images, sequentially generated 80/20 random splits, and 384-pixel short-edge
  resize before 224 crops.
- `config/optimizer/adamW.yaml`: AdamW with learning rate 3e-4 and weight decay
  1e-2.
- `config/scheduler/cosineAnnealingLR.yaml`: cosine decay after every optimizer
  step, `T_max=1888`, and zero minimum learning rate.
- `config/loss/default.yaml` and `src/utils/loss.py`: LoDa's normalized
  two-term PLCC loss.
- `src/dataset/dataloader.py`: random horizontal/vertical flips for training;
  random crops for both training and testing; ImageNet normalization.
- `src/tools/test_model.py`: average 15 patch predictions per image, compute
  raw SRCC, and compute PLCC/RMSE after four-parameter logistic fitting.

The benchmark script runs split indices 0 through 9 and the paper reports the
median SRCC and PLCC. The released code evaluates every epoch; the paper's
supplement specifies reporting the last epoch rather than selecting the best
test epoch.

The current VNCT development screen intentionally runs only split 0 once on
LIVE, LIVEC, CSIQ, and KonIQ-10k.  Each dataset is run once under
`new_modules_only` and once under `full`, for eight runs total.  This is an
engineering screen, not the paper-level ten-split protocol.

| Dataset | Split unit | Train patches | Train/test images | T_max | Local |
| --- | --- | ---: | ---: | ---: | --- |
| KonIQ-10k | image | 3 | 8,058 / 2,015 | 1,888 | ready |
| KADID-10k | reference | 3 | 8,100 / 2,025 | 1,898 | ready |
| SPAQ | image | 3 | 8,900 / 2,225 | 2,085 | missing |
| LIVEC | image | 5 | 930 / 232 | 363 | ready |
| LIVE | reference | 5 | 615 / 164 | 240 | ready |
| CSIQ | reference | 5 | 694 / 172 | 271 | ready |
| TID2013 | reference | 10 | 2,400 / 600 | 1,875 | ready |
| FLIVE | image | 1 | 31,848 / 7,962 | 2,488 | ready |

The local validator checks every YAML constant, metadata row count, and data
path:

```bash
python tools/validate_loda_configs.py
```

SPAQ has a complete expected-layout config but is marked unavailable because
there is no SPAQ directory under `/home/ssd/IQA`. LIVE and TID2013 reproduce
the released preprocessing script's use of an unsorted Python `set` for
reference IDs; exact published membership requires LoDa's distributed split
pickle because set order may vary with Python hash state.  For the current
single LIVE run, references are sorted before the seed-3407 shuffle so the
615/164 membership is reproducible across machines.

Two mappings are intentionally project-specific. LoDa's ViT-B/ResNet-50 pair
is replaced by the VSSD-Small/NC-M3 backbone, while its frozen-backbone policy
maps to training only the NC-M3 adapter and the future quality head. The local
SCQ data store contains KonIQ JPEGs at 1024x768 but not LoDa's generated
512x384 PNG directory, so the same short-edge resize to 384 is applied at
runtime before cropping.
