# CONTEXT.md — bayesnf_jax reasoning trail

This document records the full reasoning behind every architectural and
implementation decision in the bayesnf_jax project. It exists so that a
new Claude Code session (or a human collaborator) can understand not just
what was decided but why, and can make consistent decisions when extending
the codebase.

This was produced from a design conversation and should be read alongside
CLAUDE.md, which contains the actionable implementation instructions.

---

## Background and motivation

### The sibling project: jungle3

The `jungle3` project trains a BayesNF model (Google's Bayesian Neural Field
library, built on JAX) to predict per-glacier per-year glacier mass balance
from climate and topographic features. It uses one model per RGI region.
The BayesNF library performed better than plain BNNs, deep ensembles, and
MC dropout in prior experiments.

The BayesNF training loop in jungle3 is straightforward supervised training:
per-glacier per-year OGGM mass balance estimates as targets, standard VI
ELBO with BayesNF's built-in observation models.

### The new requirement: aggregated observational supervision

The motivating observation dataset (from `oggm_combined_loss.py`) demonstrates
a need to train on aggregated rather than point-level observations:
- Hugonnet (2021): per-glacier mass balance averages over multiple periods
  (2000-2010, 2000-2020, 2010-2020); finetuning uses only the single longest
  period (2000-2020) — decadal periods have higher uncertainties and hurt performance
- GLaMBIE: annual regional mass balance aggregates from gravimetry and/or
  altimetry, available for some regions and years only

Training directly on these aggregated observations with a plain BNN leads to
equifinality: many different per-glacier-per-year prediction surfaces can
produce the same aggregate loss. The model is underdetermined at the point
level.

### Why Bayesian continual learning

The solution is two-stage training:
1. Pretrain on OGGM point-level estimates to anchor the solution space
2. Finetune on observational aggregates, using the pretrained posterior as
   the prior (Bayesian continual learning)

The KL term in the finetuning ELBO then serves dual purpose:
- Regularises weights (standard ELBO role)
- Acts as a continuity constraint: the finetuned posterior cannot move
  far from what OGGM taught the model without paying a KL cost

This is preferable to MAP pretraining + fine-tuning because it preserves
calibrated uncertainty throughout both stages and explicitly encodes
"how much should we trust the new observations over the OGGM prior."

---

## Why we reimplemented rather than using BayesNF

BayesNF's VI implementation fixes the likelihood to its built-in observation
models (NORMAL, NB, etc.). The finetuning likelihood is structurally different:
it is defined over aggregated predictions, not point predictions, and involves
segment_sum operations over glacier and year groupings. There is no way to
express this as a BayesNF observation model without rewriting library internals.

Additionally, Bayesian continual learning requires access to the weight
posterior (means + log-variances) after pretraining, to use as the finetuning
prior. BayesNF does not expose this.

Therefore: both stages are implemented in a custom Flax/Optax loop, preserving
the neural field architecture (Fourier time encoding + MLP) from BayesNF but
with full control over the ELBO and likelihood.

---

## Architecture decisions

### Why Fourier encoding only on time, not all inputs

Several options were considered:
1. Fourier encoding on time only, all other features concatenated directly
   (BayesNF style)
2. Fourier encoding on the full input vector (RBF kernel approximation)
3. Factored Fourier encoding: separate encodings for time, lat/lon, and
   direct pass-through for climate/geometry features

Option 2 was rejected because the inputs are heterogeneous (periodic time,
spatial coordinates, non-periodic continuous climate variables, discrete-ish
geometry variables). An isotropic RBF kernel treats them as commensurable,
which is physically wrong.

Option 3 was discussed but the user chose Option 1, citing that the BayesNF
concatenation style already outperformed alternatives and there is no reason
to deviate from a working formulation.

The performance advantage of BayesNF over plain BNNs is attributed primarily
to the Fourier time encoding giving the network a richer periodic basis for
the time dimension, not to other aspects of BayesNF's architecture.

### Why fixed (non-learned) Fourier frequencies

Following BayesNF convention. Learned frequencies add optimisation complexity
and risk of the frequencies collapsing to degenerate solutions. Fixed random
frequencies approximate the RBF kernel in expectation over the random draw,
which is the theoretical justification for random Fourier features (Rahimi &
Recht 2007).

### Temporal index convention

Temporal index = year - T_MIN where T_MIN = 1941.

This was chosen because:
- Training data spans 2000-2019 (OGGM/Hugonnet/GLaMBIE)
- Historical inference spans 1941-present
- Future scenarios extend to ~2100
- Defining the index relative to T_MIN=1941 ensures consistent representation
  of all time points across training and inference
- Using raw year integers was rejected because BayesNF uses timetype='index'
  (integer index, not datetime) and redefining the index relative to the
  training set would make future years out-of-distribution

Fourier frequencies are log-uniformly spaced to cover periods from 2 years
(Nyquist for annual data) up to T_RANGE=159 years (full temporal range).

### Spatial coordinates

CenLat and CenLon are used directly as input features (no Fourier encoding,
no GP prior over space). The spatial structure is learned implicitly by the
MLP. This is the same approach as BayesNF and was not changed because the
prior experiments showed adequate spatial generalisation.

### Mean-field VI rather than other Bayesian approaches

Options considered: MC dropout, deep ensembles, full-covariance VI,
mean-field VI.

MC dropout and deep ensembles were rejected because they performed worse
than BayesNF in prior experiments.

Full-covariance VI was rejected as computationally intractable for the
parameter counts involved.

Mean-field VI was chosen because:
- It is what BayesNF uses under the hood (the ensemble_size parameter in
  BayesNF draws multiple weight samples from the VI posterior at prediction
  time, exactly as mc_predict() does here)
- It gives calibrated uncertainty that propagates correctly through the
  segment_sum aggregation operations in the finetuning loss
- The weight posterior (mu, log_sigma) is cheap to store and the KL against
  any Gaussian prior is analytic

---

## Loss design decisions

### Why three separate likelihood terms rather than one

OGGM (pretraining), Hugonnet, and GLaMBIE operate at different levels of
aggregation and come from different data sources with different uncertainty
characteristics. Keeping them separate allows:
- Independent uncertainty weighting per term
- Graceful handling of missing data (GLaMBIE absent for some regions)
- Clear attribution of gradient signal to each data source during debugging

### Unit standardisation: why MWE/yr throughout

All three likelihood terms are expressed in MWE/yr (metres water equivalent
per year):
- OGGM outputs in mm/yr → divide by 1000
- Hugonnet dmdtda in kg/m²/yr → divide by 1000 (1 kg/m² = 1 mm w.e.)
- GLaMBIE in regional Gt/yr sum → divide by N_glaciers to get regional mean
  MWE/yr (approximation valid when glacier area distribution is roughly
  uniform, which is acceptable for a loss term)

Converting to Gt was considered for Hugonnet (to avoid small-glacier
dominance) but rejected because:
- Inverse-variance weighting already handles small-glacier downweighting
  (small glaciers with high uncertainty get low weight regardless of units)
- Consistent MWE/yr units across all terms avoids the risk of scale
  mismatches in the ELBO
- The Gt conversion requires area² propagation of Hugonnet's per-unit-area
  uncertainty, which introduces additional complexity and potential for error

### Why GLaMBIE gravimetry and altimetry are separate residuals

When both gravimetry and altimetry are available for the same year and region,
they are treated as two independent observations rather than merged into one.
This lets the model see both constraints and is simpler to implement (no
inverse-variance merging step needed before the loss). The normalisation
denominator is N_obs = total number of (year, source) pairs, not N_years,
so years with both sources do not get double-counted in the normalisation.

### Why losses are normalised by number of residuals

Without normalisation, Hugonnet would dominate the finetuning ELBO purely
because it has N_glaciers residuals (hundreds per region) while GLaMBIE
has at most N_years * 2 residuals (≤40). Normalising each term by its
number of residuals ensures each data source contributes equally to the
ELBO in expectation, and gives the KL weight beta a consistent meaning
across terms.

### Why glambie_weight=2.0 and kl_weight=0.5, not 1.0/1.0

The theoretically "neutral" setting is glambie_weight=1.0, kl_weight=1.0:
equal per-observation weight between GLaMBIE and Hugonnet, and standard-
strength Bayesian continual learning (full KL anchor to the OGGM-pretrained
posterior). The repo does not use this setting.

The reason: OGGM's own mass-balance model calibration is itself fit against
Hugonnet geodetic mass-balance data. This means the OGGM-pretrained posterior
(and therefore the Stage 2 KL prior) already carries a Hugonnet-derived
signal, baked in during Stage 1. If Stage 2 then applies a full-strength KL
anchor (kl_weight=1.0) *and* fits `L_temporal_avg` directly to the same
Hugonnet period, Hugonnet's influence is counted twice — once indirectly
through the prior, once directly through the likelihood — while GLaMBIE
(the only Stage 2 source OGGM never saw) would be relatively underweighted
by comparison.

The fix applied: kl_weight=0.5 discounts the prior's pull (reducing the
duplicated Hugonnet-via-OGGM signal), and glambie_weight=2.0 upweights
GLaMBIE to compensate, since GLaMBIE is the one genuinely independent
observational source in Stage 2.

This is a blunt, not surgical, correction: kl_weight is a single scalar over
the entire KL term, so it discounts all of OGGM's contribution uniformly —
including the parts that are *not* duplicated with Hugonnet (the physical
mass-balance model structure, climate sensitivity learned over the full
pretrain window, and extrapolation behaviour in years/regions Hugonnet
doesn't cover). A more precise fix would restrict pretrain_year_min/max to
years where the OGGM-calibration/Hugonnet overlap is weakest, isolating the
double-counted portion rather than discounting the whole prior. This was
considered and explicitly deferred — the blanket kl_weight=0.5 discount is
judged sufficient for current purposes, and pretrain windowing is left as a
separate, independent ablation axis (see the pretrain-year sweep in
README.md) rather than a fix for this specific issue.

### Beta annealing

Beta is annealed from 0 to 1 over the first ~20% of training epochs in
both stages. This prevents posterior collapse early in training, where
the KL term would otherwise dominate before the likelihood has had a
chance to pull the posterior toward the data. This is standard practice
for VAE-style training and is especially important here because the
initial posterior (tight around the prior) produces poor predictions
that generate large likelihood gradients.

### Hugonnet: dmdtda vs dmdt

Hugonnet provides both:
- dmdt: mass change rate in Gt/yr (area-integrated)
- dmdtda: mass change rate in kg/m²/yr (area-normalised, equivalent to MWE/yr)

dmdtda was chosen because:
- The model predicts in MWE/yr, so no area conversion is needed
- Hugonnet's per-glacier uncertainty in dmdtda form already reflects
  measurement uncertainty without being inflated by area errors
- Small glaciers with genuinely uncertain measurements have large err_dmdtda
  and are naturally downweighted by inverse-variance weighting

---

## Data pipeline decisions

### Pre-split CSV structure (inherited from jungle3)

For each region, the OGGM data is pre-split into:
- train.csv: training data
- logo.csv: leave-one-glacier-out validation
- loyo.csv: leave-one-year-out validation
- loygo.csv: leave-one-glacier-year-out validation
- for_preds.csv: full grid for generating predictions
- full.csv: complete dataset (train + all validation)

This structure is inherited from jungle3 and not changed.

### One model per RGI region

The decision to train one model per region rather than a single global model
was inherited from jungle3 and not revisited. The main practical implication
is that the segment_sum aggregation in the GLaMBIE loss is exact (all glaciers
in the training data for a region correspond exactly to the GLaMBIE regional
aggregate) rather than an approximation.

### Hugonnet data format expected

Input file: dmdtda_hugo.csv per region, extracted from the full hugonnet_dmdt.csv
using extract_hugonnet_region() in data_utils.py.
Required columns: rgiid, period ('YYYY-MM-DD_YYYY-MM-DD'), dmdtda, err_dmdtda
The file may contain multiple periods (2000-2010, 2000-2020, 2010-2020).
finetune.py selects only the longest period at runtime (by comparing end_date - start_date).
Correlated glaciers (is_cor=True) are excluded by default — these are glaciers with
no direct geodetic observation for a given period (Hugonnet interpolated them).
Glacier alignment with factorized OGGM codes is handled per-period by pd.factorize
inside prepare_finetune_arrays().

### GLaMBIE data format expected

Input file: r{nn}_glambie.csv
Required columns: year (int), source ('gravimetry' or 'altimetry'),
                  mass_balance_mean_mwe (regional mean MWE/yr, pre-converted
                  from Gt/yr by dividing by N_glaciers),
                  err_mwe (uncertainty, same scaling)
Years present in this file must be a subset of years in training data.
Missing files or empty files for a region/source must be handled gracefully.

---

## Implementation notes

### segment_sum pattern (from oggm_combined_loss.py)

The aggregation pattern uses jax.ops.segment_sum with factorized integer
codes from pd.factorize(). A counter array of ones is segment-summed in
parallel to convert sums to means. This pattern must stay inside the JIT
boundary — do not use pandas groupby inside the loss function.

The factorize step happens once outside the training loop (not inside the
loss function or train_step) to avoid recompilation.

### pretrained_params.pkl format

Stored as a tuple: (mu_dict, log_sigma_dict)
where both dicts have the same pytree structure as the Flax params dict,
containing only the *_mu and *_log_sigma leaves respectively.
Produced by extract_vi_params() in bnf_module.py.
Do NOT store the full optimizer state or model object — only the posterior
parameters needed to define the Stage 2 prior.

### VIDense parameter naming

Flax stores parameters by the names passed to self.param(). VIDense uses:
- w_mu, w_log_sigma for weight parameters
- b_mu, b_log_sigma for bias parameters

The extract_vi_params() function relies on these naming conventions.
Do not rename these parameters without updating extract_vi_params().

---

## Uncertainty aggregation — spatial and temporal correlation fixes

This section documents a set of fixes made to how ensemble uncertainty
(structural, epistemic, aleatoric) is aggregated across glaciers, models, and
years in the downstream ensemble/plotting scripts (not the training pipeline —
no changes were made to pretrain.py, finetune.py, bnf_module.py, or predict.py's
core prediction logic; predict.py already wrote everything these fixes need).

### The core problem

Structural and epistemic uncertainty both arise from a fixed, shared function
(a trained model's weights, or a small set of trained models) evaluated at
many different inputs (glaciers, years). A model's bias/deviation is a smooth,
deterministic function of its inputs — not independent noise resampled per
row — so it is strongly correlated across glaciers with similar covariates,
and fully persistent across years for a fixed model (the same weights generate
the whole 86-year trajectory). Treating these components as independent when
aggregating across glaciers, models, or years (i.e. combining via quadrature,
`sqrt(sum(sigma_i^2))`) discards positive covariance terms and understates the
true aggregate uncertainty. Aleatoric uncertainty, by contrast, is a
defensible independence case (it aims to capture local/point noise), so it
keeps its quadrature (shrinking) treatment throughout.

### Category 1 — glacier -> region -> global spatial aggregation (fixed)

`src/plot_global_from_glaciers.py::load_glacier_gt()` used to re-derive a
region's contribution to the global total by summing per-glacier variance
(`groupby("year").sum()` of `(std_component * area)^2`), assuming glaciers'
epistemic/structural deviations are independent of each other. They aren't —
all glaciers in a region share the same trained model(s).

Fix: retired that per-glacier re-derivation. `build_global()` now reads each
region's already-correct `ensemble_regional_gt.csv` (produced by
`ensemble_uncertainty_pretrain_year.py` / `ensemble_uncertainty.py` /
`ensemble_ep_alea.py`, where structural/epistemic are already correctly
propagated per-model via `predict.py::compute_regional_series`'s per-MC-draw
area-weighted mean, before being combined across models via the exact
law-of-total-variance formula) and sums across regions via quadrature. That
region-to-region quadrature sum is valid — unlike glacier-to-glacier, region-
to-region correlation is not a concern, because each region has its own
independently-trained model ensemble (one-model-per-RGI-region design).
`load_regional_gt()` is now the single, only aggregation path (previously it
existed only as a "no features file" fallback for r19).

`plot_global_multi_config.py` imports `load_glacier_gt`/`load_regional_gt_fallback`
directly from `plot_global_from_glaciers.py` and inherited this fix
automatically — no separate change was needed there (that file has since been
moved to `tmp_cleanup/` for unrelated reasons — see the cleanup section of
CLAUDE.md/README).

### Category 2 — cumulative (year-to-year) aggregation (fixed)

Every cumulative-Gt plot in the repo computed `cum_std = sqrt(cumsum(std_total**2))`
— an independent-years quadrature sum applied to `std_total`, which bundles
structural, epistemic, and aleatoric together. This is wrong for the
structural/epistemic portion for the reason above. Measured on one real
86-year regional test case (r13/combined, 2026-09), the correct treatment
gave a cumulative uncertainty band ~6.8x wider than the naive formula (±72.5 Gt
vs. ±495 Gt at final year, against a cumulative median of -540 Gt) — not
enough to make the estimate meaningless, but enough that the naive band was
substantially overconfident.

Fix: `src/cumulative_uncertainty.py` (new, dependency-free module — shared by
everything below without creating a circular import with
`ensemble_uncertainty.py`) computes four cumulative-uncertainty scenarios per
site:

  - `cum_std_total` (recommended): structural = exact persistent (from the K
    ensemble members' own regional Gt trajectories — cumsum each member's own
    trajectory, then take the weighted variance across members' cumulative
    trajectories at each year; not an approximation), epistemic = persistent
    (linear sum of per-year epistemic std — this one IS an approximation,
    since exact epistemic persistence would need raw per-draw MC samples,
    which are not saved anywhere in this pipeline; epistemic is consistently
    the smallest of the three components so this approximation has limited
    practical effect), aleatoric = independent (quadrature sum, unchanged).
  - `cum_std_structural_independent`: structural forced to the naive
    independent (quadrature) treatment instead, epistemic/aleatoric as above
    — a sensitivity check on how much the structural treatment matters.
  - `cum_std_all_correlated`: aleatoric also forced to persistent — an upper
    bound, everything treated as fully correlated across years.
  - `cum_std_all_independent`: everything independent — equivalent to the old
    pre-fix formula, kept as a direct before/after comparison, not as a
    recommendation.

The exact structural term requires the K ensemble members' source multirun
run directories (via `run_dir` in `top_runs_info.csv`/`top_models_info.csv`)
to still exist (e.g. on `/scratch`) so their own `regional_annual_gt.csv` can
be re-read. If they've been cleaned up, `compute_cumulative_components()`
falls back to the independent (naive) structural treatment with a printed
warning — in that fallback case `cum_std_total == cum_std_structural_independent`.
This means a past ensemble whose source run directories have since been
deleted cannot be retroactively fixed without re-running `predict` for those
models (not a full retrain — the trained `.pkl` weights plus predict are
enough, since `predict.py` never needed to change).

Sites updated: `ensemble_uncertainty.py`, `ensemble_ep_alea.py`,
`ensemble_uncertainty_pretrain_year.py` (via the shared `_run_top_n_group` in
the new `src/ensemble_common.py`), `validate_hma.py` (computes the exact
structural term per-region for r13/r14/r15 separately, then quadrature-sums
across regions before re-deriving the four scenarios — same region-
independence reasoning as Category 1), `plot_model_animations.py`, and
`plot_acceleration_analysis.py`. `plot_global_from_glaciers.py::compute_blocks`
already did the right persistent/independent split for its 20-year block
means and was left unchanged — it's the reference implementation the other
sites' formulas are modeled on.

`plot_model_animations.py` and `plot_acceleration_analysis.py` use the
simpler "persistent" (not exact) treatment for structural too, rather than
retrieving each region's own top-N model trajectories: `anim_cumulative()`'s
uncertainty band was previously *never invoked* by `main()` despite being
loaded (`gt_std` was computed but not passed to `anim_cumulative()` calls) —
fixed to be substantively correct and actually wired up, but plumbing full
per-region model provenance into a decorative racing-animation path wasn't
judged worth the additional complexity. Both scripts save the site's usual
output plus (where a CSV/PNG is naturally produced) the 4-scenario CSV/plot;
the two animation-adjacent scripts skip the 4-panel sensitivity PNG since it
doesn't fit an animation-style deliverable.

Every fixed site additionally writes:
  - a cumulative CSV with all four scenario columns (e.g.
    `ensemble_cumulative_gt.csv`, `top_models_cumulative_gt.csv`,
    `hma_cumulative_gt.csv`) — not just the plotted band;
  - a 4-panel sensitivity PNG (`plot_cumulative_sensitivity()`), one subplot
    per scenario, shared y-axis scale for direct visual comparison, for
    every site except the two racing-animation outputs.

### Per-glacier temporal aggregation (deferred, not fixed)

A different, unaddressed gap: rigorously computing a per-glacier multi-year
average with correlation-aware uncertainty (e.g. the Hugonnet-period scatter
in `predict.py::plot_hugonnet_scatters`, which currently just averages the
per-row epistemic `std` across the period's years per glacier) needs
per-glacier raw MC samples across the relevant year window — a fundamentally
different, glacier-scoped artifact from the regional per-draw samples used
above (building the regional series already collapses/marginalizes the
glacier axis via area-weighting, so it can't be recovered afterward).

Storage checked empirically (2026-09): Hugonnet temporal-avg coverage is not
a small subset — it covers ~88% of a region's glaciers (measured on r19:
2,413 of 2,752 glaciers). Restricting to the Hugonnet window only (~20 years,
not the full record) for one selected model across all 19 regions would cost
~1.7 GB; for a full 86-year per-glacier record across the top-N ensemble,
tens of GB. This was deferred by explicit decision (the user wants the full
86-year capability eventually, not the cheaper 20-year-window version, and
declined to build the cheaper partial version now) — not fixed, and not
silently ignored: flagged inline at `plot_hugonnet_scatters` and here.

---

## Fixed vs. variable area Gt conversion

Every Gt/yr series in this pipeline is derived from an MWE/yr series via
`Gt = MWE * area_km2 * 1e-3`, where `area_km2` was always a single constant
per region (from `main_features_{region}.csv`'s `Area` column, duplicated
across every year — glaciers don't actually shrink in this input data). This
adds a second, "variable area" conversion using a real, region-specific
linear area-change rate, alongside the existing fixed-area one — never
replacing it — at every site that produces a Gt series.

### Data source and derivation

Rates were fit from GLaMBIE's own `calendar_years` regional area series
(`GlaMBIE_Data_DOI_10.5904_wgms-glambie-2024-07/glambie_results_20240716/
calendar_years/{n}_{name}.csv`, one file per RGI region, annual `glacier_area`
2000-2024) — a local path (`/mnt/c/Users/.../OneDrive - University of
Bristol/...`) not reachable from the HPC/scratch environment the pipeline
actually runs in, so the fitted rates (`slope_km2_per_yr` / `intercept_km2`
from `area = slope*year + intercept`, OLS fit against each period's midpoint
year) are embedded directly as a Python dict (`_AREA_RATES`) in
`src/area_rates.py`, not read from a data file. This was originally a
`data_for_model/glambie_area_change_rates.csv`, but `data_for_model/` is
blanket-gitignored for the pipeline's actual (large, regenerable) data, and
git's ignore semantics don't allow excepting a single file out of an
already-ignored directory once the directory itself is matched by a
trailing-slash pattern (confirmed empirically — a `!data_for_model/
glambie_area_change_rates.csv` negation rule was added and tested, and
`git check-ignore` still reported the file as ignored). Embedding the (tiny,
19-row, fixed) table directly avoids the packaging problem entirely.

**Important interpretive note**: every region's fit has R² = 1.000. This is
not us fitting a noisy independent measurement — GLaMBIE's own
`glacier_area` series is already a perfectly linear function of year (almost
certainly a two-point interpolation/extrapolation between RGI inventory
snapshots in GLaMBIE's own methodology, not an independently-measured yearly
area). So "the linear rate" is exactly the rate GLaMBIE's methodology
already assumes — describe it as "GLaMBIE-prescribed", not "our estimate".

### Extrapolation range and why no floor/clamp was needed

The fit window is 2000-2024; the pipeline's actual prediction range is
~1940-2025 (extended to at most 2050 by a separate "quadratic extension"
analysis in `outputs/paper_figures/task3_continuation/`, which is out of
scope here — see below). Checked empirically before implementing: naively
extrapolating every region's linear fit out to 2100 sends r16 (Low
Latitudes) area negative (crosses zero ~2085) and gets r11/r18 uncomfortably
close to zero — but within this pipeline's actual range (through 2050), every
region stays comfortably positive (r16, the tightest case, is still ~976 km²
at 2050 against ~1700 km² in 2024). So no floor/clamp logic was implemented;
`dynamic_area_km2()` is a bare, unguarded linear extrapolation, and the
module docstring in `src/area_rates.py` states exactly why that's safe over
this pipeline's range and would not be beyond ~2085.

Backward extrapolation to 1940 is an **acknowledged limitation, not a
validated assumption**: real glacier retreat has generally accelerated over
time (a step change around the 1980s-1990s in most regions), so applying a
constant 2000-2024 rate back to 1940 likely overstates early-decade area
loss. This is exactly why the fixed-area and variable-area series are always
shown side by side rather than the variable-area one replacing the fixed-area
default — the true historical value is expected to lie somewhere between the
two, not to equal either.

### Where this was wired in (no retraining — same MWE series, different area multiplier)

Purely post-hoc, same pattern as the uncertainty-aggregation fixes above:
every site already had an MWE series and a constant `total_area_km2` (or
equivalent); a second Gt series was added using
`area_rates.variable_area_scale(region, years)` instead, plus a
fixed-vs-variable comparison plot (`area_rates.plot_fixed_vs_variable_area`).

- `src/inference/predict.py` — `regional_annual_gt_variable_area.csv`
  (finetune and pretrain), `regional_gt_area_comparison[_pretrain].png`.
- `src/ensemble_common.py` (`_run_top_n_group`, used by
  `ensemble_uncertainty_pretrain_year.py`), `src/ensemble_uncertainty.py` —
  `ensemble_regional_gt_variable_area.csv`, `ensemble_regional_gt_area_comparison.png`.
- `src/ensemble_ep_alea.py` — `top_models_regional_gt_variable_area.csv`,
  `top_models_regional_gt_area_comparison.png`.
- `src/plot_global_from_glaciers.py` — `build_global()`/`load_regional_gt()`
  take a `filename` parameter so the same summation code produces both the
  fixed-area and variable-area global series; `global_from_glaciers_area_comparison.png`.
- `src/validate_hma.py` — `combine_hma()` takes the same `filename`
  parameter; combines r13/r14/r15's variable-area series the same way as the
  fixed-area one; `hma_area_comparison.png`.
- `src/plot_model_animations.py` — derives variable-area Gt directly from the
  already-loaded per-region MWE series (no new file dependency) for a single
  static comparison plot (`global_cumulative_area_comparison.png`); a full
  animated fixed-vs-variable comparison was judged not worth the added
  rendering complexity for a decorative racing-animation product.
- `src/plot_acceleration_analysis.py` — `load_global_gt()`/
  `load_global_gt_uncertainty()` take the same `filename` parameter;
  `acceleration_area_comparison.png`.

All variable-area code paths degrade gracefully (print a warning and skip
the comparison) when the variable-area CSV doesn't exist yet for a given
output directory — e.g. ensembles built before this feature was added.

### Explicitly out of scope for this feature

`outputs/paper_figures/` (`common_draws.py`, the Task 1-3 headline-results
series including `task3_continuation/make_continuation.py`'s quadratic
extension to 2035/2050) was **not** touched — the user explicitly said to
ignore that whole directory for this work ("I'll deal with quadratic
later"). That pipeline also does its own MWE-to-Gt conversion with a
similarly-static area assumption and has its own, separately-confirmed
correlation-structure rules for structural vs. epistemic/aleatoric
uncertainty that differ from (in fact partly invert) the rules implemented
in `src/cumulative_uncertainty.py` above — if that pipeline is revisited
later, do not assume the `src/` rules transfer to it without re-confirming.
