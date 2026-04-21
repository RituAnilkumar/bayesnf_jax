"""
Explainability entry point — Integrated Gradients attributions for one region.

Three input modes (highest priority first):

  ensemble_dir   Region multirun folder containing numbered subdirectories (0/, 1/, …).
                 Each subdir must contain finetuned_params.pkl (or pretrained_params.pkl)
                 and a .hydra/config.yaml.  Architecture and data paths are read from
                 the first valid subdir's Hydra config — no model= override needed.
                 Attributions are computed across all ensemble members, capturing both
                 VI uncertainty (n_mc_per_model draws per run) and structural uncertainty
                 (variation across the ensemble).

  run_dir        Single run directory (e.g. multirun/r01_3977063/0/).
                 Reads .hydra/config.yaml for architecture + data paths automatically.
                 No model= override needed.

  params_path    Explicit path to a single pkl file.
                 Requires model=bnf_regional_seasonal/r01 for architecture config.

Usage:
    # Ensemble mode — full structural + VI uncertainty, no model= needed:
    python main_explain.py explain.ensemble_dir=/path/to/multirun/r01_3977063

    # Single run from HPC — no model= needed:
    python main_explain.py explain.run_dir=/path/to/multirun/r01_3977063/0

    # Explicit pkl — still needs model= for architecture:
    python main_explain.py model=bnf_regional_seasonal/r01 \\
        explain.params_path=/path/to/finetuned_params.pkl

    # Waterfall for a specific glacier-year:
    python main_explain.py explain.ensemble_dir=/path/to/multirun/r01_3977063 \\
        explain.waterfall_rgi_id=RGI60-01.00001 explain.waterfall_year=2010
"""

import os
import re
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


# ---------------------------------------------------------------------------
# Param / scaler loading
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
# Architecture helpers
# ---------------------------------------------------------------------------

def _get_arch_key(params: dict):
    """
    Extract a (n_layers, nhidden, out_features) fingerprint from params shapes.

    Handles both plain hidden_layers_N and remat(hidden_layers_N) key names
    produced by nn.remat in BayesianNeuralField.

    Returns None if the structure cannot be parsed.
    """
    try:
        p = params["params"]
        n_layers, nhidden = 0, None
        while True:
            # nn.remat wraps the key name; try both forms
            for key in (f"remat(hidden_layers_{n_layers})", f"hidden_layers_{n_layers}"):
                if key in p:
                    nhidden = int(p[key]["w_mu"].shape[-1])
                    n_layers += 1
                    break
            else:
                break   # no more hidden layers found
        out_features = int(p["output_layer"]["w_mu"].shape[-1])
        return (n_layers, nhidden, out_features)
    except (KeyError, TypeError, IndexError, AttributeError):
        return None


def _build_bnf_model(mc, arch_key: tuple) -> "BayesianNeuralField":
    """Build a BayesianNeuralField from a (n_layers, nhidden, out_features) arch key."""
    n_layers, nhidden, out_features = arch_key
    return BayesianNeuralField(
        hidden_sizes=tuple([nhidden] * n_layers),
        n_fourier=int(mc.n_fourier),
        heteroscedastic=(out_features == 2),
        sigma_floor=float(mc.get("sigma_floor", 0.05)),
    )


# ---------------------------------------------------------------------------
# Hydra config loading from a run directory
# ---------------------------------------------------------------------------

def _load_run_cfg(run_dir: str) -> DictConfig:
    """Load the saved Hydra config from a run directory's .hydra/config.yaml."""
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No .hydra/config.yaml found in {run_dir}\n"
            "Make sure you are pointing at a directory produced by a Hydra run."
        )
    return OmegaConf.load(cfg_path)


# ---------------------------------------------------------------------------
# Ensemble model weighting
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
    """
    Load per-run metrics and compute model weights via build_results_df +
    add_composite_score + compute_weights (mirrors ensemble_uncertainty.py).

    Returns a (n_runs,) weights array, or None if weighting=='equal' or loading fails.
    """
    if weighting == "equal":
        return None

    try:
        from src.hyperparam_tuning import build_results_df, add_composite_score
        from src.ensemble_uncertainty import compute_weights
    except ImportError as e:
        warnings.warn(f"[explain] Cannot import weighting helpers: {e} — falling back to equal weights.")
        return None

    # build_results_df needs a single Path pointing at the parent of the run dirs
    # (all run_dirs must share the same parent).
    parents = {os.path.dirname(d) for d in run_dirs}
    if len(parents) != 1:
        warnings.warn("[explain] Ensemble members span multiple parents — equal weights used.")
        return None

    ensemble_root = Path(next(iter(parents)))
    try:
        df = build_results_df(ensemble_root, test_years, min_runs_per_region=min_runs)
        df = add_composite_score(df, loyo_weight=loyo_weight, glambie_weight=glambie_weight)
    except Exception as e:
        warnings.warn(f"[explain] build_results_df failed: {e} — equal weights used.")
        return None

    valid = df.dropna(subset=["composite"]).copy().reset_index(drop=True)
    if len(valid) == 0:
        warnings.warn("[explain] No runs with complete metrics — equal weights used.")
        return None

    # Map composite scores back to the ordered run_dirs list
    run_id_to_composite = dict(zip(valid["run_id"].values, valid["composite"].values))
    composites = []
    for d in run_dirs:
        rid = os.path.basename(d)
        composites.append(run_id_to_composite.get(rid, float("nan")))

    composites = np.array(composites)
    missing = np.sum(np.isnan(composites))
    if missing > 0:
        warnings.warn(
            f"[explain] {missing}/{len(run_dirs)} ensemble members have no metrics — "
            "those receive the worst (highest) composite score."
        )
        # Assign worst score to members with no metrics
        worst = np.nanmax(composites) if not np.all(np.isnan(composites)) else 1.0
        composites = np.where(np.isnan(composites), worst, composites)

    weights = compute_weights(composites, weighting, temperature)
    return weights


# ---------------------------------------------------------------------------
# Ensemble directory scanning
# ---------------------------------------------------------------------------

def _find_run_subdirs(ensemble_dir: str, stage: str) -> list[str]:
    """
    Return sorted list of run subdirs under ensemble_dir that contain a pkl.

    Numbered directories (0, 1, 2, …) are preferred; any subdirectory containing
    the expected pkl qualifies.
    """
    pkl_name = f"{stage}d_params.pkl"   # finetuned_params.pkl / pretrained_params.pkl
    subdirs = []
    for entry in sorted(os.scandir(ensemble_dir), key=lambda e: (
        int(e.name) if e.name.isdigit() else float("inf"), e.name
    )):
        if not entry.is_dir():
            continue
        if os.path.exists(os.path.join(entry.path, pkl_name)):
            subdirs.append(entry.path)
    if not subdirs:
        raise FileNotFoundError(
            f"No subdirectories containing '{pkl_name}' found under {ensemble_dir}"
        )
    return subdirs


# ---------------------------------------------------------------------------
# Model + data setup from a resolved config
# ---------------------------------------------------------------------------

def _build_model_and_data(run_cfg: DictConfig, orig_cwd: str):
    """
    Given a (Hydra-saved) model config, build the BNF model and load data arrays.

    Returns:
        model, time_index, covariates (original), covariates_scaled,
        rgi_ids, years, ft_cols, target_scaler, scaler
    """
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

    # Restrict to training year window to avoid far-future extrapolation artefacts
    yr_min = int(mc.get("pretrain_year_min", 2000))
    yr_max = int(mc.get("pretrain_year_max", 2020))
    features_df = features_df[features_df["year"].between(yr_min, yr_max)].copy()

    time_index, covariates, rgi_ids, cols_used = build_model_inputs(features_df, ft_cols)
    years = features_df["year"].to_numpy()

    # Return mc so the caller can rebuild the model after detecting heteroscedastic
    # from actual params (more reliable than the config for mixed-version runs).
    return mc, time_index, covariates, rgi_ids, years, cols_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config_explain", version_base=None)
def main(cfg: DictConfig) -> None:
    rng = jax.random.PRNGKey(cfg.model.seed)
    orig = get_original_cwd()
    ex = cfg.explain
    stage = ex.stage   # used for pkl filename and output naming

    # ------------------------------------------------------------------
    # Resolve absolute paths for the three source options
    # ------------------------------------------------------------------
    def _abs(p):
        return p if (p and os.path.isabs(p)) else (os.path.join(orig, p) if p else None)

    ensemble_dir = _abs(ex.get("ensemble_dir"))
    run_dir      = _abs(ex.get("run_dir"))
    params_path  = _abs(ex.get("params_path"))

    # ------------------------------------------------------------------
    # Auto-detect: if run_dir has no .hydra/config.yaml but contains
    # numbered subdirs with pkls, the user pointed at a region folder —
    # promote it to ensemble_dir automatically.
    # ------------------------------------------------------------------
    if run_dir and not ensemble_dir:
        if not os.path.exists(os.path.join(run_dir, ".hydra", "config.yaml")):
            pkl_name = f"{stage}d_params.pkl"
            has_run_subdirs = any(
                entry.is_dir() and os.path.exists(os.path.join(entry.path, pkl_name))
                for entry in os.scandir(run_dir)
                if entry.name.isdigit()
            )
            if has_run_subdirs:
                print(f"[explain] run_dir has no .hydra/config.yaml — "
                      f"detected ensemble folder, switching to ensemble mode")
                ensemble_dir = run_dir
                run_dir = None

    # ------------------------------------------------------------------
    # Determine output directory.
    # hydra.run.dir is now always "outputs/explain" (no model interpolation),
    # so we create a meaningful subdir from the source path or model config.
    # ------------------------------------------------------------------
    if ensemble_dir:
        region_tag = os.path.basename(ensemble_dir.rstrip("/"))
    elif run_dir:
        # run_dir is e.g. .../r01_3977063/0 — use the parent folder name
        region_tag = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    else:
        reg = getattr(cfg.model, "reg_subdir", "unknown")
        itype = getattr(cfg.model, "input_type", "run")
        region_tag = f"{reg}_{itype}"

    # cfg.model.output_dir is "." → resolved by Hydra chdir to outputs/explain/
    out_dir = os.path.join(cfg.model.output_dir, region_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[explain] Output dir: {os.path.abspath(out_dir)}")

    # ------------------------------------------------------------------
    # Mode 1: ensemble_dir — scan all numbered subdirs
    # ------------------------------------------------------------------
    if ensemble_dir:
        print(f"[explain] Mode: ensemble  ({ensemble_dir})")
        run_subdirs = _find_run_subdirs(ensemble_dir, stage)
        print(f"[explain] Found {len(run_subdirs)} ensemble members")

        # Architecture + data config from the first subdir
        run_cfg = _load_run_cfg(run_subdirs[0])
        mc, time_index, covariates, rgi_ids, years, cols_used = \
            _build_model_and_data(run_cfg, orig)

        # Scalers from the first subdir (all members trained with the same scaler)
        scaler         = _load_scaler(run_subdirs[0])
        target_scaler  = _load_target_scaler(run_subdirs[0])
        covariates_sc  = apply_scaler(covariates, scaler) if scaler is not None \
                         else covariates.astype(np.float32)

        # Load all params and detect full architecture from weight shapes.
        # Config values are unreliable when the multirun swept hyperparameters.
        pkl_name = f"{stage}d_params.pkl"
        all_params_raw = [
            (d, _load_params(os.path.join(d, pkl_name))) for d in run_subdirs
        ]

        arch_keys = [_get_arch_key(p) for _, p in all_params_raw]
        valid_keys = [k for k in arch_keys if k is not None]
        if not valid_keys:
            raise RuntimeError("Could not detect architecture from any ensemble member.")

        # Pick the most common architecture across all members
        arch_counts = Counter(valid_keys)
        majority_arch = arch_counts.most_common(1)[0][0]
        n_layers, nhidden, out_features = majority_arch
        heteroscedastic = (out_features == 2)

        if len(arch_counts) > 1:
            print(f"[explain] Multiple architectures detected across ensemble members:")
            for k, cnt in arch_counts.most_common():
                tag = " ← selected (majority)" if k == majority_arch else ""
                print(f"  layers={k[0]} nhidden={k[1]} out={k[2]}  ×{cnt} runs{tag}")

        compatible_dirs = []
        all_params = []
        for (d, p), key in zip(all_params_raw, arch_keys):
            if key == majority_arch:
                compatible_dirs.append(d)
                all_params.append(p)
            else:
                print(f"[explain] Skipping {os.path.basename(d)} — "
                      f"arch {key} != majority {majority_arch}")

        print(f"[explain] Using {len(all_params)}/{len(all_params_raw)} compatible ensemble members "
              f"(layers={n_layers} nhidden={nhidden} heteroscedastic={heteroscedastic})")

        # ------------------------------------------------------------------
        # Compute model weights (performance-based or equal)
        # ------------------------------------------------------------------
        weighting      = str(ex.get("weighting", "equal"))
        temperature    = float(ex.get("softmax_temperature", 0.1))
        loyo_w         = float(ex.get("loyo_weight", 0.0))
        glambie_w_cfg  = float(ex.get("glambie_weight", 1.0))
        test_years_cfg = list(ex.get("glambie_test_years") or [2021, 2022, 2023])
        min_runs_cfg   = int(ex.get("min_runs", 5))
        top_k_models   = ex.get("top_k_models")
        top_k_models   = int(top_k_models) if top_k_models else None

        ensemble_weights = _compute_ensemble_weights(
            compatible_dirs, weighting, temperature,
            loyo_w, glambie_w_cfg, test_years_cfg, min_runs_cfg,
        )

        if ensemble_weights is not None:
            print(f"[explain] Weighting='{weighting}', temperature={temperature}")
            print(f"  Weight range: [{ensemble_weights.min():.4f}, {ensemble_weights.max():.4f}]")
        else:
            print(f"[explain] Weighting='equal' ({len(all_params)} members)")

        # ------------------------------------------------------------------
        # Optional top-k filtering (applied BEFORE IG to save compute)
        # ------------------------------------------------------------------
        if top_k_models is not None and top_k_models < len(all_params):
            if ensemble_weights is not None:
                top_idx = np.argsort(ensemble_weights)[::-1][:top_k_models]
            else:
                # No scores — just take first top_k
                top_idx = np.arange(top_k_models)
            compatible_dirs = [compatible_dirs[i] for i in top_idx]
            all_params      = [all_params[i]      for i in top_idx]
            if ensemble_weights is not None:
                ensemble_weights = ensemble_weights[top_idx]
                ensemble_weights = ensemble_weights / ensemble_weights.sum()
            print(f"[explain] top_k_models={top_k_models}: reduced to {len(all_params)} members")

        model = _build_bnf_model(mc, majority_arch)

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
        label = f"{stage}_ensemble{len(all_params)}"

    # ------------------------------------------------------------------
    # Mode 2: run_dir — single run, reads its own Hydra config
    # ------------------------------------------------------------------
    elif run_dir:
        print(f"[explain] Mode: run_dir  ({run_dir})")
        run_cfg = _load_run_cfg(run_dir)
        mc, time_index, covariates, rgi_ids, years, cols_used = \
            _build_model_and_data(run_cfg, orig)

        scaler        = _load_scaler(run_dir)
        target_scaler = _load_target_scaler(run_dir)
        covariates_sc = apply_scaler(covariates, scaler) if scaler is not None \
                        else covariates.astype(np.float32)

        pkl_name  = f"{stage}d_params.pkl"
        pkl_path  = os.path.join(run_dir, pkl_name)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"{pkl_name} not found in {run_dir}")
        run_params      = _load_params(pkl_path)
        arch_key        = _get_arch_key(run_params)
        heteroscedastic = (arch_key[2] == 2) if arch_key else False
        model           = _build_bnf_model(mc, arch_key)

        mean_attrs, std_attrs, sample_idx = compute_attributions(
            model=model,
            params=run_params,
            time_index=time_index,
            covariates=covariates_sc,
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
        std_epistemic  = None
        std_structural = None
        label = stage

    # ------------------------------------------------------------------
    # Mode 3: params_path or model config fallback — needs model= on CLI
    # ------------------------------------------------------------------
    else:
        # Resolve paths that Hydra placed relative to original cwd
        cfg_mut = OmegaConf.to_container(cfg, resolve=True)
        for key in ("finetuned_params_path", "pretrained_params_path", "inp_dir"):
            if key in cfg_mut["model"] and cfg_mut["model"][key]:
                cfg_mut["model"][key] = os.path.join(orig, cfg_mut["model"][key])
        cfg = OmegaConf.create(cfg_mut)

        if params_path:
            print(f"[explain] Mode: params_path  ({params_path})")
            actual_pkl = params_path
        else:
            print(f"[explain] Mode: model config  (stage={stage})")
            actual_pkl = cfg.model.finetuned_params_path if stage == "finetune" \
                         else cfg.model.pretrained_params_path

        if not os.path.exists(actual_pkl):
            raise FileNotFoundError(
                f"Params not found: {actual_pkl}\n"
                "Use explain.run_dir or explain.ensemble_dir to point directly at "
                "a Hydra run directory and avoid relying on hardcoded config paths."
            )

        params_dir = os.path.dirname(actual_pkl)
        mc = cfg.model
        ft_cols  = list(mc.model_ftcols) if mc.get("model_ftcols") else FEATURE_COLS
        rm_fts   = list(mc.rm_fts) if mc.get("rm_fts") else []
        ft_cols  = [c for c in ft_cols if c not in rm_fts]
        region   = mc.reg_subdir
        base     = os.path.join(mc.inp_dir, region)
        features_df = load_features(
            os.path.join(base, f"main_features_{region}.csv"), feature_cols=ft_cols
        )
        yr_min = int(mc.get("pretrain_year_min", 2000))
        yr_max = int(mc.get("pretrain_year_max", 2020))
        features_df = features_df[features_df["year"].between(yr_min, yr_max)].copy()

        time_index, covariates, rgi_ids, cols_used = build_model_inputs(features_df, ft_cols)
        years = features_df["year"].to_numpy()

        loaded_params   = _load_params(actual_pkl)
        arch_key        = _get_arch_key(loaded_params)
        heteroscedastic = (arch_key[2] == 2) if arch_key else False
        model           = _build_bnf_model(mc, arch_key)

        scaler        = _load_scaler(params_dir)
        target_scaler = _load_target_scaler(params_dir)
        covariates_sc = apply_scaler(covariates, scaler) if scaler is not None \
                        else covariates.astype(np.float32)

        mean_attrs, std_attrs, sample_idx = compute_attributions(
            model=model,
            params=loaded_params,
            time_index=time_index,
            covariates=covariates_sc,
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
        std_epistemic  = None
        std_structural = None
        label = stage

    # ------------------------------------------------------------------
    # Shared: feature names + unscaled feature values for plots
    # ------------------------------------------------------------------
    feature_names = ["year"] + list(cols_used)

    fval_time = (time_index[sample_idx].reshape(-1, 1).astype(np.float32) + T_MIN)
    fval_covs = covariates[sample_idx].astype(np.float32)   # original unscaled
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
    print(f"[explain] Saved {attr_path}  ({len(attr_df)} rows)")

    # ------------------------------------------------------------------
    # Plots — helper so we can call once for full set and once for masked
    # ------------------------------------------------------------------
    # Year values from the unmasked feature_values (index 0 = year always).
    # Passed to masked dependence plots so they stay coloured by year even
    # when "year" has been removed from the feature list.
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
        # Dependence coloured by year — shows temporal drift in relationships
        plot_dependence_grid(
            attrs, fvals, fnames,
            output_dir=os.path.join(out_dir, f"dependence_year{suffix}"),
            top_k=int(ex.dependence_top_k),
            color_values=dep_color_values,   # forces year even on masked plots
        )
        # Dependence coloured by most-correlated other feature — shows interactions
        plot_dependence_grid(
            attrs, fvals, fnames,
            output_dir=os.path.join(out_dir, f"dependence_interaction{suffix}"),
            top_k=int(ex.dependence_top_k),
            color_feature=None,              # auto-select most correlated feature
        )

    # Full plots (all features) — year is in fvals so no override needed
    _generate_plots(mean_attrs, std_attrs, feature_values, feature_names, suffix="",
                    std_epi=std_epistemic, std_struct=std_structural)

    # Masked plots — exclude noisy/dominant features (year, coordinates, etc.)
    exclude = list(ex.get("exclude_features") or [])
    if exclude:
        keep_idx = [i for i, n in enumerate(feature_names) if n not in exclude]
        if keep_idx:
            dropped = [n for n in feature_names if n in exclude]
            print(f"[explain] Generating masked plots  (excluding: {dropped})")
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

    # Waterfall for a specific glacier-year
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
            print(f"WARNING: {wf_rgi} year={wf_year} not found in sampled data — "
                  "try increasing explain.max_points or choosing a different example.")

    # ------------------------------------------------------------------
    # Temporal importance evolution (always generated)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Year-slice importance bars
    # ------------------------------------------------------------------
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

    print("[explain] Done.")


if __name__ == "__main__":
    main()
