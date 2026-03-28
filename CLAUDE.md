# CLAUDE.md — bayesnf_jax project instructions

This file is the persistent instruction set for Claude Code working on the
`bayesnf_jax` project. Read this before writing any code. Read CONTEXT.md
for the full reasoning trail behind every decision here.

---

## What this project is

A two-stage Bayesian Neural Field (BNF) pipeline for predicting glacier mass
balance, trained per RGI region. It replaces the BayesNF library used in the
sibling project `jungle3` with a fully custom Flax/JAX implementation that
supports a custom aggregated observational likelihood and Bayesian continual
learning across two training stages.

---

## Folder structure

```
bayesnf_jax/
├── main_pretrain.py
├── main_finetune.py
├── main_predict.py
├── CLAUDE.md
├── CONTEXT.md
│
├── conf/
│   ├── config_pretrain.yaml
│   ├── config_finetune.yaml
│   ├── config_predict.yaml
│   └── model/
│       ├── bnf_seasonal.yaml
│       ├── bnf_monthly.yaml
│       └── bnf_seasonal/
│           ├── r01.yaml
│           ├── r02.yaml
│           └── ...
│
├── fin_runs/
│   ├── pretrain_r01_seasonal.sh
│   ├── finetune_r01_seasonal.sh
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── bnf_module.py       # ALREADY WRITTEN — see below
│   │   ├── pretrain.py         # Stage 1 training loop
│   │   └── finetune.py         # Stage 2 training loop
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── elbo.py             # ELBO assembly, beta annealing
│   │   ├── likelihood.py       # OGGM, Hugonnet, GLaMBIE likelihood terms
│   │   └── aggregation.py      # segment_sum helpers, unit conversions
│   └── inference/
│       ├── __init__.py
│       └── predict.py          # MC forward passes, quantile extraction
│
├── data_for_model/
│   ├── oggm/
│   │   └── r{nn}/
│   │       ├── train.csv
│   │       ├── logo.csv
│   │       ├── loyo.csv
│   │       ├── loygo.csv
│   │       ├── for_preds.csv
│   │       └── full.csv
│   ├── temporal_avg/
│   │   └── temporal_avg_targets_r{nn}.csv  # cols: rgi_id, start_date, end_date, avg_mb_mwe, uncertainty_mwe
│   └── glambie/
│       └── glambie_targets_r{nn}.csv       # wide format: region, start_date, end_date,
│                                           #   combined_gt/errors, altimetry_gt/errors, gravimetry_gt/errors
│                                           #   (suffix _gt is historical; values are MWE/yr)
│
└── outputs/
    ├── pretrain/
    │   └── r{nn}_{inputtype}/
    │       ├── pretrained_params.pkl
    │       ├── training_loss.png
    │       └── metrics.csv
    └── finetune/
        └── r{nn}_{inputtype}/
            ├── finetuned_params.pkl
            ├── preds_full.csv
            ├── preds_quantiles.csv
            ├── training_loss.png
            └── metrics.csv
```

---

## What is already written

### `src/model/bnf_module.py`
The core architecture module. Contains:
- `FourierTimeEncoder` — fixed random Fourier features for the time coordinate
- `VIDense` — dense layer with mean-field VI (reparameterisation trick)
- `BayesianNeuralField` — full model: Fourier encoder + VI MLP
- `kl_gaussian` — analytic KL for diagonal Gaussians
- `compute_total_kl` — total KL over all model parameters
- `extract_vi_params` — splits params dict into mu and log_sigma sub-dicts
- `make_standard_normal_prior` — constructs N(0,1) prior matching params structure

Do not rewrite this file unless there is a bug. Extend it only if new
architectural components are needed.

---

## Architecture decisions — do not deviate without good reason

### Fourier time encoding
- Applied to time coordinate ONLY (not spatial or climate features)
- Follows BayesNF convention: fixed (non-learned) random frequency matrix
- Frequencies log-uniformly spaced between 1/159 and 1/2 cycles/yr
- Default n_fourier=8 (BayesNF default)
- Temporal index = year - 1940 (T_MIN=1940, T_MAX=2100, T_RANGE=160)
- This index must be consistent across training, historical inference,
  and future scenario prediction — never redefine it relative to training data

### Spatial and covariate inputs
- CenLat, CenLon passed directly (no Fourier encoding, no GP prior)
- Climate features (tas, pr, rsds variants) passed directly
- Glacier geometry (Area, Zmed, Slope, Aspect, Zmin, Zmax) passed directly
- All concatenated with Fourier time features before the MLP
- This is faithful to the BayesNF concatenation style that outperformed
  plain BNN, deep ensembles, and MC dropout in prior experiments

### Mean-field VI
- Every weight and bias has learned mu and log_sigma
- Forward pass samples weights via reparameterisation: w = mu + exp(log_sigma)*eps
- Initial log_sigma = -3.0 so initial posterior is tight (sigma ≈ 0.05)
- At inference: mc_predict() draws N independent weight samples via vmap

### One model per RGI region
- Do not attempt a single multi-region model
- Each region has its own pretrained_params.pkl and finetuned_params.pkl

---

## Training pipeline

### Stage 1 — Pretrain on OGGM (main_pretrain.py)
- Data: per-glacier per-year mass balance in MWE/yr from OGGM
  (OGGM outputs in mm/yr — divide by 1000 before use)
- Splits available: train, logo, loyo, loygo, for_preds, full
  (same structure as jungle3/data_for_model/)
- Likelihood: mean over all (glacier × year) residuals
  L_oggm = (1 / N_points) * sum_ij (pred_ij - oggm_ij)²
- Prior: N(0, 1) — use make_standard_normal_prior()
- KL: KL(posterior || N(0,1)), analytic
- ELBO = L_oggm - beta * KL
- Beta annealing: anneal beta from 0 → 1 over first ~20% of epochs
  (prevents posterior collapse early in training)
- Output: save (mu_dict, log_sigma_dict) as pretrained_params.pkl
  using cloudpickle — this becomes the prior for Stage 2

### Stage 2 — Finetune on observations (main_finetune.py)
- Load pretrained_params.pkl as the prior (Bayesian continual learning)
- Prior: N(mu_pre, sigma_pre) — NOT standard normal
- KL: KL(finetuned posterior || pretrained posterior), analytic
- Three likelihood terms, all in MWE/yr:

  L_temporal_avg = (1/N_glaciers) * sum_i [
      (pred_period_mean_i - avg_mb_mwe_i)² / uncertainty_mwe_i²
  ]
  where:
  - pred_period_mean_i = mean of per-year predictions for glacier i over
    start_date–end_date (typically 2001-2020), via segment_sum / count_sum
  - avg_mb_mwe_i, uncertainty_mwe_i are already in MWE/yr — no unit conversion

  L_glambie = (1/N_obs) * sum_k [
      (pred_ann_mean_t(k) - glambie_mean_t(k))² / err_glambie_mean_t(k)²
  ]
  where:
  - k indexes all available (year, source) pairs — gravimetry and altimetry
    are treated as separate residuals
  - pred_ann_mean_t = mean of per-glacier predictions in year t
    (regional sum via segment_sum, divided by N_glaciers)
  - glambie_mean_t = GLaMBIE regional Gt/yr sum divided by N_glaciers
    to convert to MWE/yr mean (err scaled by same factor)
  - N_obs = total number of (year, source) observations, NOT unique years
    (so years with both gravimetry and altimetry count as 2)

  ELBO = L_temporal_avg + L_glambie - beta * KL(finetuned || pretrained)

- No manual loss weighting beyond uncertainty weighting and unit normalisation
- GLaMBIE may be absent for some regions (e.g. High Mountain Asia has no
  gravimetry) — handle gracefully, do not error if glambie file is missing
  or empty for a given source

### Inference (main_predict.py)
- Load finetuned_params.pkl
- Draw N samples from posterior via mc_predict()
- Run over full (glacier × year) grid: 1941–present + future scenarios
- Report 2.5th, 50th, 97.5th percentiles of predictive distribution
- Output: preds_full.csv and preds_quantiles.csv

---

## Config conventions (Hydra)

Follow the same Hydra override pattern as jungle3:
- One config file per input type (seasonal, monthly)
- Per-region overrides via fin_runs/ SLURM scripts
- Key model config params:
  model.inp_dir, model.reg_subdir
  model.model_nlayers, model.model_nhidden, model.n_fourier
  model.model_nepochs, model.model_nensemble (= N MC samples at inference)
  model.model_ftcols, model.rm_fts
  model.pretrained_params_path  (Stage 2 only)
  model.temporal_avg_path       (Stage 2 only)
  model.glambie_path            (Stage 2 only)
  model.beta_anneal_epochs      (epochs over which beta is annealed 0→1)

---

## Data conventions

- OGGM targets: mm/yr → divide by 1000 → MWE/yr before any loss computation
- Temporal avg avg_mb_mwe: already MWE/yr — no conversion
- Temporal avg uncertainty_mwe: already MWE/yr — no conversion
- GLaMBIE: values already MWE/yr (column suffix _gt is historical, ignore it)
- GLaMBIE source selection: prefer altimetry + gravimetry; combined is fallback
  only when both altimetry and gravimetry are NaN for a given row
- Year column in OGGM data may be int (annual) or date string (seasonal) —
  handle both as in jungle3/src/model/bayesnf_oggm.py
- Temporal avg period is 2001-2020 per glacier (start_date/end_date columns)
- GLaMBIE source values (after load_glambie reshape): 'altimetry', 'gravimetry', 'combined'
- GLaMBIE year derived from floor(end_date) — end_date is a fractional year

---

## Key invariants — always check these

1. Temporal-avg rgi_id ordering must match the segment_sum gid_codes ordering
   (use pd.factorize on the training data, then assert alignment with
   temporal_avg — same pattern as oggm_combined_loss.py)
2. GLaMBIE years must be a subset of the years present in the training data
3. Temporal index (year - 1941) must be applied consistently in all data
   loading, training, and inference code — never use raw year integers as
   model input
4. Beta annealing must complete (beta=1.0) before the end of training in
   both stages — do not let beta remain < 1 at convergence
5. pretrained_params.pkl stores only (mu_dict, log_sigma_dict) as a tuple —
   not the full Flax model or optimizer state
