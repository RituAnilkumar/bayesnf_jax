"""
Explainability entry point — Integrated Gradients attributions for one region.

Loads a trained model (finetuned or pretrained), computes per-feature
Integrated Gradients attributions for a chosen data split, saves the results
as CSVs, and produces four plot types:
  - importance_bar.png       — global mean |attribution| per feature
  - beeswarm.png             — distribution of attributions across data points
  - dependence_{feat}.png    — attribution vs feature value for top-k features
  - waterfall_{id}_{yr}.png  — local attribution for a single glacier-year

Usage:
    python main_explain.py model=bnf_regional_seasonal/r01
    python main_explain.py model=bnf_regional_seasonal/r01 explain.stage=pretrain
    python main_explain.py model=bnf_regional_seasonal/r01 explain.n_mc=50 explain.max_points=5000
    python main_explain.py model=bnf_regional_seasonal/r01 \\
        explain.waterfall_rgi_id=RGI60-01.00001 explain.waterfall_year=2010
"""

import os
import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from src.model.bnf_module import BayesianNeuralField, T_MIN
from src.data_utils import (
    load_features,
    build_model_inputs,
    apply_scaler,
    FEATURE_COLS,
)
from src.explainability import (
    compute_attributions,
    plot_importance_bar,
    plot_beeswarm,
    plot_dependence_grid,
    plot_waterfall,
)


# ---------------------------------------------------------------------------
# Helpers — reuse param loading from predict.py
# ---------------------------------------------------------------------------

def _load_params(path: str) -> dict:
    with open(path, "rb") as f:
        result = cloudpickle.load(f)
    assert isinstance(result, tuple) and len(result) == 2
    mu_dict, log_sigma_dict = result

    def _merge(d_mu, d_ls):
        out = {**d_mu}
        for k, v in d_ls.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return {"params": _merge(mu_dict, log_sigma_dict)}


def _load_scaler(params_dir: str):
    path = os.path.join(params_dir, "scaler.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def _load_target_scaler(params_dir: str):
    path = os.path.join(params_dir, "target_scaler.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config_explain", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.model.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(cfg.model.seed)
    orig = get_original_cwd()

    # Resolve relative paths
    cfg_mut = OmegaConf.to_container(cfg, resolve=True)
    for key in ("finetuned_params_path", "pretrained_params_path", "inp_dir"):
        if key in cfg_mut["model"] and cfg_mut["model"][key]:
            cfg_mut["model"][key] = os.path.join(orig, cfg_mut["model"][key])
    cfg = OmegaConf.create(cfg_mut)

    ex = cfg.explain
    stage = ex.stage   # 'finetune' or 'pretrain'

    # --- Select params path ---
    if stage == "pretrain":
        params_path = cfg.model.pretrained_params_path
    else:
        params_path = cfg.model.finetuned_params_path
    params_dir = os.path.dirname(params_path)

    print(f"[explain] Stage: {stage}  |  region: {cfg.model.reg_subdir}")
    print(f"[explain] Params: {params_path}")

    # --- Feature columns ---
    ft_cols = list(cfg.model.model_ftcols) if cfg.model.model_ftcols else FEATURE_COLS
    rm_fts  = list(cfg.model.rm_fts) if cfg.model.get("rm_fts") else []
    ft_cols = [c for c in ft_cols if c not in rm_fts]

    # --- Load data ---
    region  = cfg.model.reg_subdir
    base    = os.path.join(cfg.model.inp_dir, region)

    split = ex.explain_split
    split_file = os.path.join(base, f"main_features_{region}.csv")
    if not os.path.exists(split_file):
        # Fall back to old naming
        split_file = os.path.join(base, f"{split}.csv")
    features_df = load_features(split_file, feature_cols=ft_cols)

    # Filter to a sensible year window to avoid far-future extrapolation artefacts
    features_df = features_df[features_df["year"].between(
        cfg.model.get("pretrain_year_min", 2000),
        cfg.model.get("pretrain_year_max", 2020),
    )].copy()

    time_index, covariates, rgi_ids, cols_used = build_model_inputs(features_df, ft_cols)
    years = features_df["year"].to_numpy()

    # --- Apply feature scaler (same as training) ---
    scaler = _load_scaler(params_dir)
    if scaler is not None:
        covariates_scaled = apply_scaler(covariates, scaler)
    else:
        covariates_scaled = covariates.astype(np.float32)
        print("WARNING: no scaler.pkl found — using raw covariate values")

    target_scaler = _load_target_scaler(params_dir)

    # --- Load params ---
    params = _load_params(params_path)

    # --- Build model ---
    heteroscedastic = bool(cfg.model.get("heteroscedastic", False))
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
        heteroscedastic=heteroscedastic,
        sigma_floor=float(cfg.model.get("sigma_floor", 0.05)),
    )

    # --- Feature names for plots: year + covariate names ---
    feature_names = ["year"] + list(cols_used)

    # --- Compute attributions ---
    print(f"[explain] Computing IG attributions: n_mc={ex.n_mc}, n_steps={ex.n_steps}, "
          f"baseline={ex.baseline}, max_points={ex.max_points}")

    mean_attrs, std_attrs, sample_idx = compute_attributions(
        model=model,
        params=params,
        time_index=time_index,
        covariates=covariates_scaled,
        rng=rng,
        n_mc=int(ex.n_mc),
        n_steps=int(ex.n_steps),
        heteroscedastic=heteroscedastic,
        baseline=ex.baseline,
        target_scaler=target_scaler,
        max_points=int(ex.max_points),
        chunk_size=int(ex.chunk_size),
        seed=int(ex.seed),
    )

    # Covariate feature values for colour encoding — use (unscaled) originals
    # for interpretable axes; scaled values are fine for beeswarm colouring
    fval_time = time_index[sample_idx].reshape(-1, 1).astype(np.float32) + T_MIN  # actual year
    fval_covs = covariates[sample_idx].astype(np.float32)  # original unscaled
    feature_values = np.concatenate([fval_time, fval_covs], axis=1)  # (N', n_features)

    # --- Save attributions CSV ---
    attr_mean_cols = {f"attr_{n}": mean_attrs[:, i] for i, n in enumerate(feature_names)}
    attr_std_cols  = {f"std_{n}":  std_attrs[:, i]  for i, n in enumerate(feature_names)}
    attr_df = pd.DataFrame({
        "rgi_id": rgi_ids[sample_idx],
        "year":   years[sample_idx],
        **attr_mean_cols,
        **attr_std_cols,
    })
    attr_path = os.path.join(cfg.model.output_dir, f"attributions_{stage}.csv")
    attr_df.to_csv(attr_path, index=False)
    print(f"[explain] Saved {attr_path}  ({len(attr_df)} rows)")

    # --- Global importance bar ---
    plot_importance_bar(
        mean_attrs, feature_names,
        output_path=os.path.join(cfg.model.output_dir, f"importance_bar_{stage}.png"),
        std_attrs=std_attrs,
        top_k=ex.get("importance_top_k") or None,
    )

    # --- Beeswarm ---
    plot_beeswarm(
        mean_attrs, feature_values, feature_names,
        output_path=os.path.join(cfg.model.output_dir, f"beeswarm_{stage}.png"),
        n_max=int(ex.beeswarm_n_max),
        seed=int(ex.seed),
    )

    # --- Dependence plots (top-k features) ---
    plot_dependence_grid(
        mean_attrs, feature_values, feature_names,
        output_dir=cfg.model.output_dir,
        top_k=int(ex.dependence_top_k),
    )

    # --- Waterfall for a specific glacier-year ---
    wf_rgi  = ex.get("waterfall_rgi_id")
    wf_year = ex.get("waterfall_year")

    if wf_rgi and wf_year:
        wf_year = int(wf_year)
        mask = (rgi_ids[sample_idx] == wf_rgi) & (years[sample_idx] == wf_year)
        if mask.any():
            row = int(np.where(mask)[0][0])
            safe_id = wf_rgi.replace("/", "-")
            plot_waterfall(
                mean_attrs[row], feature_names,
                output_path=os.path.join(
                    cfg.model.output_dir, f"waterfall_{safe_id}_{wf_year}_{stage}.png"
                ),
                std_attr_1d=std_attrs[row],
                title=f"Local attribution — {wf_rgi}  {wf_year}  [{stage}]",
            )
        else:
            print(f"WARNING: rgi_id={wf_rgi} year={wf_year} not found in the sampled data — "
                  "try increasing max_points or choosing a different example.")

    print("[explain] Done.")


if __name__ == "__main__":
    main()
