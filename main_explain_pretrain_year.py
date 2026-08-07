"""
Explainability entry point for pretrain-year-split ensemble outputs.

Variant of main_explain.py for ensembles produced by
ensemble_uncertainty_pretrain_year.py, which writes into nested subdirectories:

    {ensemble_base_dir}/{region}/pt1940/
    {ensemble_base_dir}/{region}/pt1960/
    {ensemble_base_dir}/{region}/pt1980/
    {ensemble_base_dir}/{region}/pt2000/
    {ensemble_base_dir}/{region}/combined/

Each subfolder contains top_runs_info.csv with a run_dir column pointing at the
original multirun run directories (where finetuned_params.pkl lives).

Requires one extra config key:
    explain.pretrain_year_group: "pt2000"   # or pt1940, pt1960, pt1980, combined

Usage:
    # Single pretrain-year group:
    python main_explain_pretrain_year.py \\
        explain.ensemble_dir=outputs/ensemble_pretrain_year/r01 \\
        explain.pretrain_year_group=pt2000

    # Combined ensemble (attributions pooled across all pretrain years):
    python main_explain_pretrain_year.py \\
        explain.ensemble_dir=outputs/ensemble_pretrain_year/r01 \\
        explain.pretrain_year_group=combined

    # With explicit config:
    python main_explain_pretrain_year.py \\
        --config-name config_explain_pretrain_year \\
        explain.ensemble_dir=outputs/ensemble_pretrain_year/r01 \\
        explain.pretrain_year_group=combined
"""

import os
import warnings
import cloudpickle
from collections import Counter
from pathlib import Path
import jax
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
    compute_ensemble_attributions,
    plot_importance_bar,
    plot_beeswarm,
    plot_dependence_grid,
    plot_waterfall,
    plot_year_slice,
    plot_temporal_importance,
)


_VALID_GROUPS = {"pt1940", "pt1960", "pt1980", "pt2000", "combined"}


# ---------------------------------------------------------------------------
# Param / scaler loading (identical to main_explain.py)
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
    p = os.path.join(params_dir, "scaler.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return cloudpickle.load(f)


def _load_target_scaler(params_dir: str):
    p = os.path.join(params_dir, "target_scaler.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Architecture helpers (identical to main_explain.py)
# ---------------------------------------------------------------------------

def _get_arch_key(params: dict):
    try:
        p = params["params"]
        n_layers, nhidden = 0, None
        while True:
            for key in (f"remat(hidden_layers_{n_layers})", f"hidden_layers_{n_layers}"):
                if key in p:
                    nhidden = int(p[key]["w_mu"].shape[-1])
                    n_layers += 1
                    break
            else:
                break
        out_features = int(p["output_layer"]["w_mu"].shape[-1])
        return (n_layers, nhidden, out_features)
    except (KeyError, TypeError, IndexError, AttributeError):
        return None


def _build_bnf_model(mc, arch_key: tuple, use_time_encoding: bool = True) -> "BayesianNeuralField":
    n_layers, nhidden, out_features = arch_key
    return BayesianNeuralField(
        hidden_sizes=tuple([nhidden] * n_layers),
        n_fourier=int(mc.n_fourier),
        heteroscedastic=(out_features == 2),
        sigma_floor=float(mc.get("sigma_floor", 0.05)),
        use_time_encoding=use_time_encoding,
    )


def _get_run_use_time_encoding(run_dir: str) -> bool:
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return True
    try:
        import yaml as _yaml
        with open(cfg_path) as fh:
            raw = _yaml.safe_load(fh)
        return bool(raw.get("model", {}).get("use_time_encoding", True))
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Hydra config loading from a run directory
# ---------------------------------------------------------------------------

def _load_run_cfg(run_dir: str) -> DictConfig:
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No .hydra/config.yaml found in {run_dir}\n"
            "Make sure you are pointing at a directory produced by a Hydra run."
        )
    return OmegaConf.load(cfg_path)


# ---------------------------------------------------------------------------
# Pretrain-year-specific: resolve run dirs from top_runs_info.csv
# ---------------------------------------------------------------------------

def _find_run_subdirs_pretrain_year(ensemble_dir: str, pretrain_year_group: str,
                                    stage: str) -> list[str]:
    """
    Load top_runs_info.csv from {ensemble_dir}/{pretrain_year_group}/ and return
    the run_dir paths listed there (original multirun dirs with pkl files).
    """
    top_runs_csv = os.path.join(ensemble_dir, pretrain_year_group, "top_runs_info.csv")
    if not os.path.exists(top_runs_csv):
        raise FileNotFoundError(
            f"top_runs_info.csv not found at {top_runs_csv}\n"
            "Make sure ensemble_uncertainty_pretrain_year.py has been run and "
            "ensemble_dir points at the per-region output folder "
            "(e.g. outputs/ensemble_pretrain_year/r01)."
        )
    df = pd.read_csv(top_runs_csv)
    if "run_dir" not in df.columns:
        raise ValueError(f"top_runs_info.csv at {top_runs_csv} has no 'run_dir' column.")

    pkl_name = f"{stage}d_params.pkl"
    run_dirs = []
    for rd in df["run_dir"].values:
        rd = str(rd)
        pkl_path = os.path.join(rd, pkl_name)
        if not os.path.exists(pkl_path):
            warnings.warn(f"[explain_py] {pkl_name} not found in {rd} — skipping.")
            continue
        run_dirs.append(rd)

    if not run_dirs:
        raise FileNotFoundError(
            f"No run dirs from {top_runs_csv} contain '{pkl_name}'. "
            "Check that the multirun output is still accessible on the filesystem."
        )
    return run_dirs


# ---------------------------------------------------------------------------
# Model + data setup from a resolved config (identical to main_explain.py)
# ---------------------------------------------------------------------------

def _build_model_and_data(run_cfg: DictConfig, orig_cwd: str, yr_max_override=None):
    mc = run_cfg.model

    ft_cols = list(mc.model_ftcols) if mc.get("model_ftcols") else FEATURE_COLS
    rm_fts  = list(mc.rm_fts) if mc.get("rm_fts") else []
    ft_cols = [c for c in ft_cols if c not in rm_fts]

    inp_dir = mc.inp_dir
    if not os.path.isabs(inp_dir):
        inp_dir = os.path.join(orig_cwd, inp_dir)

    region  = mc.reg_subdir
    base    = os.path.join(inp_dir, region)
    feat_file = os.path.join(base, f"main_features_{region}.csv")

    features_df = load_features(feat_file, feature_cols=ft_cols)

    yr_min = int(mc.get("pretrain_year_min", 2000))
    yr_max = int(yr_max_override) if yr_max_override is not None \
             else int(mc.get("pretrain_year_max", 2020))
    features_df = features_df[features_df["year"].between(yr_min, yr_max)].copy()

    time_index, covariates, rgi_ids, cols_used = build_model_inputs(features_df, ft_cols)
    years = features_df["year"].to_numpy()

    return mc, time_index, covariates, rgi_ids, years, cols_used


# ---------------------------------------------------------------------------
# Ensemble model weighting (identical to main_explain_te.py)
# ---------------------------------------------------------------------------

def _compute_ensemble_weights(
    run_dirs: list[str],
    weighting: str,
    temperature: float,
    loyo_weight: float,
    glambie_weight: float,
    test_years: list[int],
    min_runs: int,
) -> "np.ndarray | None":
    if weighting == "equal":
        return None

    try:
        from src.hyperparam_tuning import build_results_df, add_composite_score
        from src.ensemble_uncertainty import compute_weights
    except ImportError as e:
        warnings.warn(f"[explain_py] Cannot import weighting helpers: {e} — falling back to equal weights.")
        return None

    parents = {os.path.dirname(d) for d in run_dirs}
    if len(parents) != 1:
        warnings.warn("[explain_py] Ensemble members span multiple parents — equal weights used.")
        return None

    ensemble_root = Path(next(iter(parents)))
    try:
        df = build_results_df(ensemble_root, test_years, min_runs_per_region=min_runs)
        df = add_composite_score(df, loyo_weight=loyo_weight, glambie_weight=glambie_weight)
    except Exception as e:
        warnings.warn(f"[explain_py] build_results_df failed: {e} — equal weights used.")
        return None

    valid = df.dropna(subset=["composite"]).copy().reset_index(drop=True)
    if len(valid) == 0:
        warnings.warn("[explain_py] No runs with complete metrics — equal weights used.")
        return None

    run_id_to_composite = dict(zip(valid["run_id"].values, valid["composite"].values))
    composites = []
    for d in run_dirs:
        rid = os.path.basename(d)
        composites.append(run_id_to_composite.get(rid, float("nan")))

    composites = np.array(composites)
    missing = np.sum(np.isnan(composites))
    if missing > 0:
        warnings.warn(
            f"[explain_py] {missing}/{len(run_dirs)} ensemble members have no metrics — "
            "those receive the worst (highest) composite score."
        )
        worst = np.nanmax(composites) if not np.all(np.isnan(composites)) else 1.0
        composites = np.where(np.isnan(composites), worst, composites)

    from src.ensemble_uncertainty import compute_weights
    weights = compute_weights(composites, weighting, temperature)
    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config_explain_pretrain_year", version_base=None)
def main(cfg: DictConfig) -> None:
    rng = jax.random.PRNGKey(cfg.model.seed)
    orig = get_original_cwd()
    ex = cfg.explain
    stage = ex.stage

    def _abs(p):
        return p if (p and os.path.isabs(p)) else (os.path.join(orig, p) if p else None)

    ensemble_dir        = _abs(ex.get("ensemble_dir"))
    pretrain_year_group = ex.get("pretrain_year_group")

    if not ensemble_dir:
        raise ValueError("explain.ensemble_dir must be set for main_explain_pretrain_year.py")
    if not pretrain_year_group:
        raise ValueError(
            "explain.pretrain_year_group must be set "
            f"(choose one of: {sorted(_VALID_GROUPS)})"
        )
    if pretrain_year_group not in _VALID_GROUPS:
        raise ValueError(
            f"explain.pretrain_year_group={pretrain_year_group!r} is not valid. "
            f"Choose one of: {sorted(_VALID_GROUPS)}"
        )

    # Output dir: {output_dir}/{region_tag}/{pretrain_year_group}
    region_tag = os.path.basename(ensemble_dir.rstrip("/"))
    out_dir = os.path.join(cfg.model.output_dir, region_tag, pretrain_year_group)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[explain_py] Output dir: {os.path.abspath(out_dir)}")

    # ------------------------------------------------------------------
    # Ensemble mode — resolve run dirs from top_runs_info.csv
    # ------------------------------------------------------------------
    print(f"[explain_py] Mode: ensemble ({ensemble_dir} / {pretrain_year_group})")
    run_subdirs = _find_run_subdirs_pretrain_year(ensemble_dir, pretrain_year_group, stage)
    print(f"[explain_py] Found {len(run_subdirs)} ensemble members in top_runs_info.csv")

    # Architecture + data config from the first subdir
    run_cfg = _load_run_cfg(run_subdirs[0])
    yr_max_ov = ex.get("explain_year_max") or None
    mc, time_index, covariates, rgi_ids, years, cols_used = \
        _build_model_and_data(run_cfg, orig, yr_max_override=yr_max_ov)

    scaler        = _load_scaler(run_subdirs[0])
    target_scaler = _load_target_scaler(run_subdirs[0])
    covariates_sc = apply_scaler(covariates, scaler) if scaler is not None \
                    else covariates.astype(np.float32)

    pkl_name = f"{stage}d_params.pkl"
    all_params_raw = [
        (d, _load_params(os.path.join(d, pkl_name))) for d in run_subdirs
    ]

    arch_keys = [_get_arch_key(p) for _, p in all_params_raw]
    ute_flags = [_get_run_use_time_encoding(d) for d in run_subdirs]
    full_keys = [
        (ak[0], ak[1], ak[2], ute) if ak is not None else None
        for ak, ute in zip(arch_keys, ute_flags)
    ]
    valid_full_keys = [k for k in full_keys if k is not None]
    if not valid_full_keys:
        raise RuntimeError("Could not detect architecture from any ensemble member.")

    full_key_counts = Counter(valid_full_keys)
    majority_full   = full_key_counts.most_common(1)[0][0]
    majority_arch   = majority_full[:3]
    majority_ute    = majority_full[3]
    n_layers, nhidden, out_features = majority_arch
    heteroscedastic = (out_features == 2)

    if len(full_key_counts) > 1:
        print(f"[explain_py] Multiple arch+TE combinations across ensemble members:")
        for k, cnt in full_key_counts.most_common():
            tag = " ← selected (majority)" if k == majority_full else ""
            print(f"  layers={k[0]} nhidden={k[1]} out={k[2]} use_time_encoding={k[3]}  "
                  f"×{cnt} runs{tag}")

    compatible_dirs = []
    all_params = []
    for (d, p), fk in zip(all_params_raw, full_keys):
        if fk == majority_full:
            compatible_dirs.append(d)
            all_params.append(p)
        else:
            print(f"[explain_py] Skipping {os.path.basename(d)} — "
                  f"arch/TE {fk} != majority {majority_full}")

    print(f"[explain_py] Using {len(all_params)}/{len(all_params_raw)} compatible members "
          f"(layers={n_layers} nhidden={nhidden} heteroscedastic={heteroscedastic} "
          f"use_time_encoding={majority_ute})")

    # ------------------------------------------------------------------
    # Compute model weights
    # ------------------------------------------------------------------
    weighting     = str(ex.get("weighting", "equal"))
    temperature   = float(ex.get("softmax_temperature", 0.1))
    loyo_w        = float(ex.get("loyo_weight", 0.0))
    glambie_w_cfg = float(ex.get("glambie_weight", 1.0))
    test_years_cfg = list(ex.get("glambie_test_years") or [2021, 2022, 2023])
    min_runs_cfg  = int(ex.get("min_runs", 5))
    top_k_models  = ex.get("top_k_models")
    top_k_models  = int(top_k_models) if top_k_models else None

    ensemble_weights = _compute_ensemble_weights(
        compatible_dirs, weighting, temperature,
        loyo_w, glambie_w_cfg, test_years_cfg, min_runs_cfg,
    )

    if ensemble_weights is not None:
        print(f"[explain_py] Weighting='{weighting}', temperature={temperature}")
        print(f"  Weight range: [{ensemble_weights.min():.4f}, {ensemble_weights.max():.4f}]")
    else:
        print(f"[explain_py] Weighting='equal' ({len(all_params)} members)")

    # Optional top-k filtering
    if top_k_models is not None and top_k_models < len(all_params):
        if ensemble_weights is not None:
            top_idx = np.argsort(ensemble_weights)[::-1][:top_k_models]
        else:
            top_idx = np.arange(top_k_models)
        compatible_dirs  = [compatible_dirs[i] for i in top_idx]
        all_params       = [all_params[i] for i in top_idx]
        if ensemble_weights is not None:
            ensemble_weights = ensemble_weights[top_idx]
            ensemble_weights = ensemble_weights / ensemble_weights.sum()
        print(f"[explain_py] top_k_models={top_k_models}: reduced to {len(all_params)} members")

    model = _build_bnf_model(mc, majority_arch, use_time_encoding=majority_ute)

    mean_attrs, std_attrs, std_epistemic, std_structural, sample_idx = \
        compute_ensemble_attributions(
            model=model,
            all_params=all_params,
            time_index=time_index,
            covariates=covariates_sc,
            rng=rng,
            n_mc_per_model=int(ex.n_mc_per_model),
            n_steps=int(ex.n_steps),
            heteroscedastic=heteroscedastic,
            baseline=ex.baseline,
            target_scaler=target_scaler,
            max_points=int(ex.max_points),
            chunk_size=int(ex.chunk_size),
            seed=int(ex.seed),
            weights=ensemble_weights,
        )
    label = f"{stage}_ensemble{len(all_params)}_{pretrain_year_group}"

    # ------------------------------------------------------------------
    # Feature names + unscaled feature values for plots
    # ------------------------------------------------------------------
    feature_names = ["year"] + list(cols_used)

    fval_time = (time_index[sample_idx].reshape(-1, 1).astype(np.float32) + T_MIN)
    fval_covs = covariates[sample_idx].astype(np.float32)
    feature_values = np.concatenate([fval_time, fval_covs], axis=1)

    # ------------------------------------------------------------------
    # Save attributions CSV
    # ------------------------------------------------------------------
    attr_df = pd.DataFrame(
        {"rgi_id": rgi_ids[sample_idx], "year": years[sample_idx]},
    )
    for i, n in enumerate(feature_names):
        attr_df[f"attr_{n}"] = mean_attrs[:, i]
        attr_df[f"std_{n}"]  = std_attrs[:, i]

    attr_path = os.path.join(out_dir, f"attributions_{label}.csv")
    attr_df.to_csv(attr_path, index=False)
    print(f"[explain_py] Saved {attr_path}  ({len(attr_df)} rows)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    year_color_values = feature_values[:, 0] if "year" in feature_names else None

    def _generate_plots(attrs, std, fvals, fnames, suffix,
                        std_epi=None, std_struct=None, dep_color_values=None):
        plot_importance_bar(
            attrs, fnames,
            output_path=os.path.join(out_dir, f"importance_bar_{label}{suffix}.png"),
            std_attrs=std,
            top_k=ex.get("importance_top_k") or None,
            std_epistemic_attrs=std_epi,
            std_structural_attrs=std_struct,
        )
        plot_beeswarm(
            attrs, fvals, fnames,
            output_path=os.path.join(out_dir, f"beeswarm_{label}{suffix}.png"),
            n_max=int(ex.beeswarm_n_max),
            seed=int(ex.seed),
        )
        plot_dependence_grid(
            attrs, fvals, fnames,
            output_dir=os.path.join(out_dir, f"dependence_year{suffix}"),
            top_k=int(ex.dependence_top_k),
            color_values=dep_color_values,
        )
        plot_dependence_grid(
            attrs, fvals, fnames,
            output_dir=os.path.join(out_dir, f"dependence_interaction{suffix}"),
            top_k=int(ex.dependence_top_k),
            color_feature=None,
        )

    _generate_plots(mean_attrs, std_attrs, feature_values, feature_names, suffix="",
                    std_epi=std_epistemic, std_struct=std_structural)

    exclude = list(ex.get("exclude_features") or [])
    if exclude:
        keep_idx = [i for i, n in enumerate(feature_names) if n not in exclude]
        if keep_idx:
            dropped = [n for n in feature_names if n in exclude]
            print(f"[explain_py] Generating masked plots  (excluding: {dropped})")
            _generate_plots(
                mean_attrs[:, keep_idx],
                std_attrs[:, keep_idx],
                feature_values[:, keep_idx],
                [feature_names[i] for i in keep_idx],
                suffix="_masked",
                std_epi=std_epistemic[:, keep_idx] if std_epistemic is not None else None,
                std_struct=std_structural[:, keep_idx] if std_structural is not None else None,
                dep_color_values=year_color_values,
            )

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
                    out_dir, f"waterfall_{safe_id}_{wf_year}_{label}.png"
                ),
                std_attr_1d=std_attrs[row],
                title=f"Local attribution — {wf_rgi}  {wf_year}  [{label}]",
            )
            if exclude and keep_idx:
                plot_waterfall(
                    mean_attrs[row][keep_idx], [feature_names[i] for i in keep_idx],
                    output_path=os.path.join(
                        out_dir, f"waterfall_{safe_id}_{wf_year}_{label}_masked.png"
                    ),
                    std_attr_1d=std_attrs[row][keep_idx],
                    title=f"Local attribution (masked) — {wf_rgi}  {wf_year}  [{label}]",
                )
        else:
            print(f"WARNING: {wf_rgi} year={wf_year} not found in sampled data.")

    plot_temporal_importance(
        mean_attrs, years[sample_idx], feature_names,
        output_path=os.path.join(out_dir, f"temporal_importance_{label}.png"),
        top_k=int(ex.dependence_top_k),
    )
    if exclude and keep_idx:
        plot_temporal_importance(
            mean_attrs[:, keep_idx],
            years[sample_idx],
            [feature_names[i] for i in keep_idx],
            output_path=os.path.join(out_dir, f"temporal_importance_{label}_masked.png"),
            top_k=int(ex.dependence_top_k),
        )

    slice_years = list(ex.get("year_slice_years") or [])
    for yr in slice_years:
        yr = int(yr)
        suffix = f"_{yr}"
        plot_year_slice(
            mean_attrs, years[sample_idx], feature_names,
            target_year=yr,
            output_path=os.path.join(out_dir, f"importance_year{suffix}_{label}.png"),
            std_epistemic_attrs=std_epistemic,
            std_structural_attrs=std_structural,
            top_k=ex.get("importance_top_k") or None,
        )
        if exclude and keep_idx:
            plot_year_slice(
                mean_attrs[:, keep_idx],
                years[sample_idx],
                [feature_names[i] for i in keep_idx],
                target_year=yr,
                output_path=os.path.join(out_dir, f"importance_year{suffix}_{label}_masked.png"),
                std_epistemic_attrs=std_epistemic[:, keep_idx] if std_epistemic is not None else None,
                std_structural_attrs=std_structural[:, keep_idx] if std_structural is not None else None,
                top_k=ex.get("importance_top_k") or None,
            )

    print("[explain_py] Done.")


if __name__ == "__main__":
    main()
