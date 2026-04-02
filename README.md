# bayesnf_jax

A two-stage Bayesian Neural Field (BNF) pipeline for predicting glacier mass balance, trained per RGI region.

- **Stage 1 (pretrain)**: Trains on OGGM per-glacier per-year mass balance estimates using a variational ELBO with a standard normal prior.
- **Stage 2 (finetune)**: Bayesian continual learning using the pretrained posterior as a prior, with two observational likelihood terms: temporal-average per-glacier targets and GLaMBIE regional annual observations.
- **Inference**: Monte Carlo forward passes from the finetuned posterior to produce predictive quantiles over the full glacier × year grid.

---

## Data layout

All input data lives under `data_for_model/`. Each region has its own subdirectory `r{nn}/` (e.g. `r01`, `r06`).

```
data_for_model/
└── r{nn}/
    ├── main_features_r{nn}.csv        # glacier features + climate, one row per (glacier, year)
    ├── oggm_targets_r{nn}.csv         # OGGM mass balance targets, mm/yr
    ├── temporal_avg_targets_r{nn}.csv # per-glacier temporal average, MWE/yr
    └── glambie_targets_r{nn}.csv      # regional GLaMBIE observations, wide format
```

### Column requirements

**`main_features_r{nn}.csv`**
```
rgi_id, CenLat, CenLon, year, GLIMSId (dropped automatically),
t2m_abl_mean, t2m_abl_std, t2m_acc_mean, t2m_acc_std,
tp_abl_sum, tp_acc_sum, ssrd_abl_sum, ssrd_acc_sum,
Aspect, Zmax, Zmed, Slope, Zmin, Area
```

**`oggm_targets_r{nn}.csv`**
```
year, rgi_id, mass_balance   (units: mm/yr — converted to MWE/yr automatically)
```

**`temporal_avg_targets_r{nn}.csv`** (or `dmdtda_hugo.csv`)
```
rgi_id, start_date, end_date, avg_mb_mwe, uncertainty_mwe
```
May contain multiple Hugonnet periods (e.g. 2000-2010, 2000-2020, 2010-2020).
Finetuning automatically selects the single longest period (typically 2000–2020).

**`glambie_targets_r{nn}.csv`**
```
region, start_date, end_date,
combined_gt, combined_gt_errors,
altimetry_gt, altimetry_gt_errors,
gravimetry_gt, gravimetry_gt_errors
```
`start_date` / `end_date` are fractional years (e.g. 1999.75). Values are MWE/yr.
Altimetry and gravimetry are used preferentially; combined is a fallback only if
the entire file has zero valid primary observations.

---

## Installation

```bash
pip install jax jaxlib flax optax cloudpickle hydra-core omegaconf pandas numpy matplotlib
```

For GPU support follow the [JAX installation guide](https://github.com/google/jax#installation) before installing the rest.

---

## Running the pipeline

Region configs live in `conf/model/bnf_regional_seasonal/r{nn}.yaml`.
Edit the paths in the relevant region file before running if your data layout differs.

### Full pipeline for one region (pretrain → finetune → predict)

```bash
python main_pipeline.py model=bnf_regional_seasonal/r06
```

To include the OGGM cross-validation stage before the full pretrain:

```bash
python main_pipeline.py model=bnf_regional_seasonal/r06 pipeline.stages=[pretrain_cv,pretrain_full,finetune,predict]
```

### Individual stages

```bash
# Stage 1 — CV run (evaluates loyo/logo/loygo splits)
python main_pretrain.py model=bnf_regional_seasonal/r06 model.train_split=train

# Stage 1 — Full pretrain (produces pretrained_params.pkl for Stage 2)
python main_pretrain.py model=bnf_regional_seasonal/r06 model.train_split=full

# Stage 2 — Finetune
python main_finetune.py model=bnf_regional_seasonal/r06

# Inference
python main_predict.py model=bnf_regional_seasonal/r06
```

### Monthly input type

Replace `bnf_regional_seasonal` with `bnf_regional_monthly` in any command above.
Update `model_ftcols` in `conf/model/bnf_regional_monthly.yaml` with the actual
monthly climate column names once those CSVs are available.

---

## Outputs

Hydra writes each run into its own directory under `outputs/`:

```
outputs/
├── pretrain/
│   └── r{nn}_seasonal_train/      # CV run — contains metrics_oos.csv
│   └── r{nn}_seasonal_full/       # Full run — contains pretrained_params.pkl
├── finetune/
│   └── r{nn}_seasonal/            # contains finetuned_params.pkl, metrics_glambie_test.csv
└── predictions/
    └── r{nn}_seasonal/            # contains preds_full.csv, preds_quantiles.csv
```

**Key output files:**

| File | Description |
|------|-------------|
| `pretrained_params.pkl` | Tuple `(mu_dict, log_sigma_dict)` — Stage 1 posterior, used as Stage 2 prior |
| `finetuned_params.pkl` | Tuple `(mu_dict, log_sigma_dict)` — Stage 2 posterior |
| `metrics_oos.csv` | OGGM CV metrics: region, split, rmse, bias, n_points. `loyo` is the primary metric. |
| `metrics_glambie_test.csv` | GLaMBIE validation on post-2020 years: year, source, glambie_mwe, pred_mwe, residual |
| `preds_full.csv` | Per-(glacier, year): rgi_id, year, p2_5, p50, p97_5, mean, std [MWE/yr] |
| `preds_quantiles.csv` | Same as above, p2_5 / p50 / p97_5 only |

---

## Validation protocol

**Stage 1 (OGGM pretrain):**
- Held-out years: 10% of years, globally consistent across all regions (fixed seed)
- Held-out glaciers: 10% of glaciers, per region
- `loyo` = held years, non-held glaciers — **primary metric**
- `logo` = held glaciers, non-held years
- `loygo` = held years AND held glaciers
- `train` = neither held

**Stage 2 (finetune):**
- No held-out data from temporal_avg (longest Hugonnet period fully used)
- GLaMBIE years ≤ temporal_avg end year → training loss
- GLaMBIE years > temporal_avg end year (typically 2021–2024) → test, automatically held out by construction
- Early stopping monitors training loss only (no validation set); activates after beta annealing completes

---

## Key config parameters

All configurable in `conf/model/bnf_regional_seasonal.yaml` or overridden per-region:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_nlayers` | 2 | Number of hidden layers |
| `model_nhidden` | 64 | Hidden layer width |
| `n_fourier` | 8 | Fourier time encoding features (output: 2×n_fourier) |
| `model_nepochs` | 2000 | Training epochs per stage |
| `lr` | 1e-3 | Adam learning rate |
| `beta_anneal_epochs` | 400 | Epochs over which KL weight ramps 0→1 |
| `model_nensemble` | 100 | MC samples for evaluation and inference |
| `seed` | 0 | RNG seed (controls weight init, CV splits, MC samples) |
| `rm_fts` | `[]` | Feature columns to exclude at runtime |

---

## Adding a new region

1. Place data files in `data_for_model/r{nn}/` following the layout above.
2. Region config files are already created for r01–r19 in both
   `conf/model/bnf_regional_seasonal/` and `conf/model/bnf_regional_monthly/`.
   Verify the paths in `conf/model/bnf_regional_seasonal/r{nn}.yaml` are correct.
3. Run the pipeline as shown above.


# Ritu's Quick Runs
python main_pipeline.py -m model=bnf_regional_seasonal/r06 model.model_nlayers=1,2,3 model.model_nhidden=16,32,64  pipeline.stages=[pretrain_cv,pretrain_full,finetune,predict] 

Setup in the sh files for all regions. Run this:
for r in $(seq -w 1 19); do
  sbatch --export=ALL,REGION=r${r} --job-name=${r}_seasonal run_regs.sh 
done