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
combined_mwe, combined_mwe_errors,
altimetry_mwe, altimetry_mwe_errors,
gravimetry_mwe, gravimetry_mwe_errors
```
`start_date` / `end_date` are fractional years (e.g. 1999.75). All values are in MWE/yr.
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

---

## Typical end-to-end workflow

The usual sequence is: **multirun sweep → hyperparameter analysis → ensemble uncertainty → validation → explanations**.

### Step 1 — Multirun hyperparameter sweep (all regions, all configs)

Submit one SLURM job per region, sweeping over architecture and model flags.
`-m` triggers Hydra multirun; each combo gets its own numbered subdirectory under the multirun root.

```bash
for r in $(seq -w 1 19); do
  sbatch --export=ALL,REGION=r${r} --job-name=${r}_sweep run_sweep.sh
done
```

Where `run_sweep.sh` contains something like:

```bash
python main_pipeline.py -m \
  model=bnf_regional_seasonal/${REGION} \
  model.model_nlayers=1,2 \
  model.model_nhidden=32,64,128 \
  model.heteroscedastic=true,false \
  model.inp_dir=/scratch/.../data_for_model \
  pipeline.stages=[pretrain_cv,pretrain_full,finetune,predict]
```

Outputs land in `multirun/r{nn}_{jobid}/{combo_idx}/` with one `preds_full.csv`,
`metrics_glambie_test.csv`, and `._cv/metrics_oos.csv` per combo.

---

### Step 2 — Hyperparameter analysis

Reads all completed runs in the multirun tree and produces ranked tables + plots
(parallel coordinates, heatmaps, RMSE distributions).

```bash
# Single region
python src/hyperparam_tuning.py \
  --multirun_root /scratch/.../multirun/r06_*

# All regions (batch)
for i in $(seq -w 1 19); do
  python src/hyperparam_tuning.py \
    --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_499*/
done
```

Outputs go to `outputs/analysis/r{nn}/` (or `--output_dir`). Key files:
- `ranked_runs.csv` — all runs sorted by composite score (LOYO RMSE + GLaMBIE RMSE)
- `parallel_coords.png`, `heatmap_*.png` — visual sweep summaries
- `nhidden_vs_rmse.png` — architecture comparison split by heteroscedastic flag

---

### Step 3 — Ensemble uncertainty

Choose the script that matches what was swept:

**If the sweep included `model.heteroscedastic=true,false`** — use `ensemble_uncertainty_split.py`,
which partitions runs into `aleatoric/` (heteroscedastic=true) and `epistemic_structural/`
(heteroscedastic=false) and writes separate ensemble outputs for each group:

```bash
for i in $(seq -w 1 19); do
  python src/ensemble_uncertainty_split.py \
    --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_*/ \
    --output_dir outputs/ensemble_no_ts/r${i}
done
```

**If only heteroscedastic runs are wanted** — use `ensemble_ep_alea.py`, which selects
the top-N heteroscedastic models by GLaMBIE test RMSE and decomposes uncertainty into
epistemic, aleatoric, and structural components:

```bash
for i in $(seq -w 1 19); do
  python src/ensemble_ep_alea.py \
    --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_*/ \
    --output_dir outputs/best_model/r${i}
done
```

Both scripts write `ensemble_regional_mwe.csv`, `ensemble_regional_gt.csv`,
`ensemble_glacier.csv`, and uncertainty decomposition plots.

---

### Step 4 — Validation

**Regional validation** against WGMS / Dussaillant reference series:

```bash
python src/validate_regional.py \
  --ensemble_base_dir outputs/ensemble \
  --output_dir outputs/validation_regional
```

Produces per-region time series plots, scatter plots, residual bars,
and `validation_metrics.csv` (RMSE, bias, correlation, coverage).

**Per-glacier validation** against WGMS direct measurements:

```bash
python src/validate_per_glacier.py \
  --ensemble_base_dir outputs/best_model \
  --output_dir outputs/validation_per_glacier
```

---

### Step 5 — Explanations (Integrated Gradients)

Compute and plot per-feature attributions for one or more regions.
Pass the full multirun ensemble directory so attributions cover both VI uncertainty
and structural (model-to-model) uncertainty:

```bash
python main_explain.py \
  explain.ensemble_dir=/scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r06_12345

# To generate a waterfall plot for a specific glacier-year:
python main_explain.py \
  explain.ensemble_dir=/scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r06_12345 \
  explain.waterfall_rgi_id=RGI60-06.00001 \
  explain.waterfall_year=2010
```

---

### Full pipeline for one region (single config, no sweep)

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
- GLaMBIE years < temporal_avg end year → training loss
- GLaMBIE years > temporal_avg end year (typically 2021–2024) → test, automatically held out by construction
- Early stopping monitors training loss only (no validation set); activates after beta annealing completes

---

## Key config parameters

All configurable in `conf/model/bnf_regional_seasonal.yaml` or overridden per-region.

### Stage 1 (pretrain)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_nlayers` | 2 | Number of hidden layers |
| `model_nhidden` | 64 | Hidden layer width |
| `n_fourier` | 8 | Fourier time encoding features (output: 2×n_fourier) |
| `model_nepochs` | 100000 | Training epochs |
| `lr` | 1e-3 | Adam learning rate |
| `beta_anneal_epochs` | 2000 | Epochs over which KL weight ramps 0→1 |
| `oggm_loss_fn` | `rmse` | Point-level loss: `rmse` or `huber` |
| `huber_delta` | 0.5 | Huber transition point in scaled units (only if `oggm_loss_fn=huber`) |
| `pretrain_year_min` | 2000 | Restrict OGGM training to years ≥ this value |
| `pretrain_year_max` | 2020 | Restrict OGGM training to years ≤ this value |
| `train_split` | `full` | `full` for final model; `train` for CV run |
| `model_nensemble` | 100 | MC samples for OOS evaluation |
| `seed` | 0 | RNG seed |
| `rm_fts` | `[]` | Feature columns to exclude at runtime |

### Stage 2 (finetune)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `finetune_lr` | 3e-4 | Initial Adam LR (cosine-decayed after beta annealing) |
| `grad_clip_norm` | 2.0 | Global gradient norm clip (0 = disabled) |
| `glambie_weight` | 1.0 | GLaMBIE loss weight relative to Hugonnet temporal-avg loss |

### Inference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_nensemble` | 100 | MC samples for prediction quantiles |

---

## Adding a new region

1. Place data files in `data_for_model/r{nn}/` following the layout above.
2. Region config files are already created for r01–r19 in both
   `conf/model/bnf_regional_seasonal/` and `conf/model/bnf_regional_monthly/`.
   Verify the paths in `conf/model/bnf_regional_seasonal/r{nn}.yaml` are correct.
3. Run the pipeline as shown above.


## Quick notes for Ritu

```bash
# Hyperparameter sweep over architecture and annealing schedule:
python main_pipeline.py -m \
  model=bnf_regional_seasonal/r06 \
  model.model_nlayers=1,2,3 \
  model.model_nhidden=8,16,32,64,128 \
  model.beta_anneal_epochs=2000,10000,20000 \
  model.inp_dir=/scratch/.../data_for_model \
  pipeline.stages=[pretrain_cv,pretrain_full,finetune,predict]

# Submit all regions via SLURM:
for r in $(seq -w 1 19); do
  sbatch --export=ALL,REGION=r${r} --job-name=${r}_seasonal run_regs.sh
done
```

After all runs are complete, I do a hyperparameter tuning and ensemble generation followed by validation regional and per glacier. Fianlly there is explain:

for i in $(seq -w 1 19); do   python src/hyperparam_tuning_te.py \\;     --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_439*/; done

for i in $(seq -w 1 19); do   python src/ensemble_uncertainty_time_encoding.py --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_439*/ --output_dir outputs/ensemble_te_r2/r${i}; done

python src/validate_regional_te.py   --config conf/config_validate_regional.yaml   --te_group no_time_encoding   --ensemble_base_dir /scratch/b5at/ranil.b5at/bayesnf_jax/outputs/ensemble_te_r2   --output_dir outputs/validation_regional_no_te   --ensemble_filename ensemble_regional_mwe.csv

python src/validate_per_glacier_te.py --te_group no_time_encoding   --ensemble_base_dir /scratch/b5at/ranil.b5at/bayesnf_jax/outputs/ensemble_te_r2   --output_dir outputs/validation_pergla_te