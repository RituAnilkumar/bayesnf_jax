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
- Hugonnet (2021): 20-year per-glacier mass balance averages (2000-2019)
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

Input file: r{nn}_hugonnet.csv
Required columns: rgi_id, dmdtda, err_dmdtda
(Other Hugonnet columns — period, area, dmdt, err_dmdt — are not used)
The rgi_id ordering in this file must be aligned with the factorized
glacier codes from the training data (assert this explicitly in code).

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
