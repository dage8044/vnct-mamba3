# Dataset configs

Add one YAML file per BIQA dataset. At minimum, define the image root,
metadata CSV, image/MOS column names, and content-separated train/validation/
test split policy. Dataset-specific score orientation and range belong here,
not in the backbone config.

`koniq10k_loda.yaml` follows the official LoDa KonIQ-10k preprocessing and
10-split protocol while mapping its data locations to the server root
`/home/ssd/IQA`. The checked-in paths are examples for this server. After a
local clone, edit `root`, `source_image_root`, and `metadata` in each dataset
file you plan to run, then execute:

```bash
python tools/validate_loda_configs.py
```

The seven `*_loda.yaml` files cover every dataset in LoDa's released
`benchmark_loda_all.sh`. Six are mapped to verified local data. SPAQ is fully
specified but has `available: false` until it is placed under
`/home/ssd/IQA/SPAQ`.
