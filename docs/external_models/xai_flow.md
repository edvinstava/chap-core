# XAI Flow in CHAP

This page explains how CHAP generates explainability outputs for a prediction, in a simple end-to-end flow.

## Big picture

For each prediction, CHAP can explain results in two ways:

- **Native SHAP**: use SHAP values produced directly by the model.
- **Surrogate-based XAI**: train a separate tabular model that mimics the prediction outputs, then run SHAP or LIME on that surrogate.

Most methods in the XAI API are surrogate-based (`shap_*`, `shap_auto`, `lime`, `lime_auto`).

## End-to-end flow

1. A prediction is stored with forecast rows (`org_unit`, `period`, forecast samples).
2. The XAI endpoint is called for global, local, beeswarm, or horizon explanations.
3. CHAP loads the source dataset and builds a surrogate training matrix:
   - `X`: covariate values matched to each forecast row.
   - `y`: one scalar target per forecast row (median by default, or mean/quantile).
   - `groups`: org-unit ids for grouped cross-validation.
4. If the method is native SHAP and native SHAP data exists, CHAP reads stored values directly (no surrogate fitting).
5. Otherwise CHAP trains a surrogate model (or reuses a cached one for the same prediction/method/output statistic).
6. CHAP computes explanations:
   - **Global**: top features across rows.
   - **Local**: feature attribution for one row (`org_unit`, `period`).
   - **Beeswarm**: many local attributions packed into one response.
   - **Horizon summary**: local attributions grouped over forecast steps.
7. CHAP stores explanation outputs in `PredictionExplanation` rows (local) and prediction metadata (global), then serves them through API responses.

## How surrogate training data is built

CHAP creates one training row per forecast:

- Finds matching covariate row by `org_unit` and `period`.
- Supports period fallback logic for horizon-style periods like `YYYYMM_k`.
- Builds missing-value statistics per feature.
- Imputes missing covariates with feature medians.
- Computes `y` from forecast samples:
  - `median` (default),
  - `mean`, or
  - `qXX` quantile.

This gives the supervised dataset used for surrogate fitting.

## How SHAP surrogate models are trained

For `shap_*` methods:

1. **Feature filtering**
   - Remove constant features.
   - Remove features with very high imputation rate (>= 90%).
   - On larger datasets, optionally remove weak/noisy features with permutation importance.
2. **Model selection**
   - If method is fixed (for example `shap_xgboost`), use that model type.
   - If method is `shap_auto`, benchmark multiple supported surrogate families using leave-one-out or leave-one-group-out R², then keep top candidates.
3. **Hyperparameter tuning**
   - For enough samples, run Optuna tuning with CV.
   - For `shap_auto`, tune candidate models and keep the best tuned model.
4. **Final fit**
   - Fit the selected surrogate on filtered features.
   - Optionally apply target transformation (`log1p` or Yeo-Johnson) if CV suggests better fit.
5. **Quality scoring**
   - Compute fidelity metrics such as CV R², train R², MAE, residual stats, and fidelity tier.

After training, CHAP computes SHAP values from the surrogate and maps them back to the full feature list (removed features get zero attribution).

About perturbations:

- Tree-based SHAP on tree surrogates does not explicitly permute each feature one by one at runtime. It computes each feature's marginal contribution from tree paths and split structure in the fitted surrogate.
- For model-agnostic SHAP (for example permutation/kernel styles), perturbed or masked feature values are evaluated through the surrogate prediction function to estimate contributions.

## How LIME surrogate models are trained

For `lime` and `lime_auto`, the training path is the same as SHAP up to surrogate fitting:

1. Build the same `(X, y, groups)` data.
2. Run the same feature filtering.
3. Use the same model selection logic (`auto` chooses the best surrogate family).
4. Run the same tuning logic when data size allows.
5. Fit the final surrogate model.

The difference is only in attribution:

- **SHAP methods** use SHAP explainers on the fitted surrogate.
- **LIME methods** use `LimeTabularExplainer` around the fitted surrogate predictions to estimate local contributions by perturbing the instance neighborhood.

So SHAP and LIME share the same surrogate training pipeline; they differ in how they extract attributions from the trained surrogate.

Why LIME is not run directly on the original model:

- CHAP's prediction interfaces are heterogeneous (Python, R, Docker, remote services), and many do not expose a stable, low-latency tabular `predict` callable needed for LIME neighborhood sampling.
- LIME requires many repeated local evaluations; calling the original model repeatedly can be expensive, non-deterministic, or operationally fragile.
- The fitted surrogate provides a consistent, fast prediction surface aligned with the stored forecast target (`median`, `mean`, or `qXX`), so both SHAP and LIME operate on the same explanation target and are easier to compare.

## Caching and persistence

- Trained surrogates are cached in memory per key `(prediction_id, method, output_statistic)` to avoid repeated fits in the same process.
- Local explanations are persisted in `PredictionExplanation`.
- Global summaries are persisted in prediction metadata under method-specific keys.

This is why repeated calls for the same inputs are usually much faster.
