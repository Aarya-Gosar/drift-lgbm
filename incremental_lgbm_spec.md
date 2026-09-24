# Incremental LightGBM with Correlation-Aware Feature Importance

## Project Codename: `drift-lgbm`

---

## INSTRUCTIONS FOR CLAUDE CODE

You are building a Python library called `drift-lgbm`. Work iteratively: build one module at a time, write tests for it, verify the tests pass, then move to the next module. Do NOT scaffold the entire project at once. Each iteration should produce working, tested code.

If you hit a design decision, pick the pragmatic option and document why in a comment. Do not ask me — make the call, test it, and move on.

After each module is complete, run the full test suite to check for regressions. If something breaks, fix it before moving forward.

The build order is strict. Do not skip ahead.

---

## WHAT THIS IS

A Python library that solves two genuine problems with LightGBM in production financial ML:

**Problem 1: Incremental training that doesn't silently degrade.**
LightGBM's `init_model` lets you add trees on new data, but the histogram bins are frozen from the first batch. When feature distributions drift (which always happens in finance — prices move, volatility regimes change, volumes shift), new data gets crammed into stale bins and splitting resolution is destroyed in exactly the range that matters. Additionally, old trees from dead regimes keep voting forever.

**Problem 2: Feature importance is misleading with correlated features.**
Financial features are heavily correlated (momentum_10d vs momentum_15d, various vol measures, sector exposures). Standard SHAP/importance splits attribution arbitrarily across correlated features. Drop one "unimportant" correlated feature and the other's importance doubles. This makes feature selection unreliable.

**The combination is the point:** an incremental model that can tell you "these feature *clusters* matter now vs. last quarter" — with proper accounting for correlation — gives a portfolio manager something actionable.

---

## WHAT THIS IS NOT

- Not a fork of LightGBM's C++ core. We use LightGBM as a dependency and build system-level intelligence on top.
- Not a research paper reimplementation. The pieces draw from known ideas (Lopez de Prado's clustered importance, standard distribution drift detection) but the system combining them is new.
- Not an AutoML tool. The user controls their features, their target, their hyperparameters. We handle the incremental training and importance analysis correctly.

---

## BUILD ORDER

### Phase 1: Core Data Structures and Feature Distribution Tracker

**File:** `drift_lgbm/distribution_tracker.py`

Build a `DistributionTracker` class that:

1. Maintains a running histogram (NOT LightGBM's internal bins — our own) per feature using a fixed number of bins (default 256).
2. Uses quantile-based binning from the first batch, then tracks when the distribution has drifted enough that rebinning is needed.
3. Drift detection method: two-sample Kolmogorov-Smirnov test between the stored reference distribution and the new batch distribution. If p-value < threshold (default 0.01), flag that feature as drifted.
4. Tracks: per-feature min, max, quantiles (stored as a digest — use `numpy` percentile computations, not t-digest to avoid dependencies), sample count, and the reference distribution histogram.
5. `update(X_batch)` method that ingests a new batch and returns a `DriftReport` dataclass listing which features drifted and by how much.
6. `should_rebin()` method that returns True if >30% of features have drifted beyond threshold.

**Key design decisions:**
- Use numpy arrays internally, accept pandas DataFrames or numpy arrays as input.
- Store feature names if DataFrame is provided, otherwise use integer indices.
- The tracker must be serializable with pickle.

**Tests for Phase 1:**
- Create synthetic data where features have known distributions. Feed batch 1, then feed batch 2 with shifted mean. Verify drift is detected for shifted features and not for stable ones.
- Test that `should_rebin()` triggers correctly based on the fraction threshold.
- Test serialization round-trip.
- Edge cases: single feature, constant feature (zero variance), NaN handling (should ignore NaNs in distribution computation).

---

### Phase 2: Tree Scorer and Retirement Manager

**File:** `drift_lgbm/tree_manager.py`

Build a `TreeManager` class that:

1. Takes a trained LightGBM Booster and evaluates the contribution of each tree to current performance.
2. **Tree scoring method:** For a validation set (X_val, y_val), compute the prediction with all trees, then compute the prediction with each tree removed (leave-one-out). The tree's contribution is the change in loss (e.g., RMSE for regression, log-loss for classification). Trees that increase performance when removed have negative contribution.
3. **Optimization:** Full leave-one-out is O(n_trees × n_samples). For large ensembles, use an approximation: since each tree's prediction is additive, the full prediction is `sum(tree_i(X))`. Removing tree_j gives prediction `full_pred - tree_j(X)`. This is O(n_trees × n_samples) but with a small constant — just subtract each tree's contribution from the full prediction. This should be fast.
4. `score_trees(X_val, y_val, metric)` returns a list of (tree_index, contribution_score) sorted by contribution.
5. `identify_retiring(threshold)` returns indices of trees whose contribution is below threshold (default: 0, meaning they hurt performance).
6. `retire_trees(booster, indices)` creates a new Booster with those trees removed. **Implementation:** Use LightGBM's `model_to_string()` / `model_from_string()` to parse the model text format, remove the specified tree blocks, update the tree count, and reconstruct. This is hacky but it works — document it as such.

**Key design decisions:**
- Support both regression (RMSE, MAE) and binary classification (log-loss, AUC) metrics.
- The `retire_trees` method returns a NEW booster, never mutates in place.
- Set a floor: never retire more than 50% of trees in one call (safety guardrail).

**Tests for Phase 2:**
- Train a LightGBM model on clean data. Add 10 "poison" trees by training `init_model` on random noise data. Verify that `score_trees` ranks the poison trees at the bottom.
- Verify that `retire_trees` produces a valid Booster that can still predict.
- Verify that retiring negatively-contributing trees improves or maintains validation performance.
- Test the 50% floor guardrail.

---

### Phase 3: Correlation-Aware Feature Importance

**File:** `drift_lgbm/feature_importance.py`

Build a `ClusteredImportance` class that:

1. **Feature clustering:** Compute a correlation matrix (Spearman rank, not Pearson — financial features have nonlinear relationships). Convert to a distance matrix: `distance = 1 - |correlation|`. Apply agglomerative clustering with a distance threshold (default: 0.5, meaning features with |correlation| > 0.5 are clustered together). Use `scipy.cluster.hierarchy`.

2. **Cluster-level importance:** For each cluster, compute importance as the sum of the raw importances (gain-based from LightGBM) of all features in the cluster. This is the cluster's total contribution.

3. **Intra-cluster attribution:** Within each cluster, report each feature's share of the cluster importance. This is informational, not for feature selection — the point is that you should select/drop at the cluster level, not the individual feature level.

4. **Stability analysis:** Given importance snapshots from multiple incremental training batches, compute the rank correlation (Kendall's tau) of cluster importances across time. Clusters with unstable importance are unreliable signals. Return a stability score per cluster.

5. **Output format:** A `ClusteredImportanceReport` dataclass with:
   - `clusters`: dict mapping cluster_id → list of feature names
   - `cluster_importance`: dict mapping cluster_id → importance score
   - `feature_importance`: dict mapping feature_name → (cluster_id, intra_cluster_share)
   - `stability_scores`: dict mapping cluster_id → stability score (only if multiple snapshots provided)
   - `correlation_matrix`: the Spearman correlation matrix used
   - A `summary()` method that prints a human-readable report.

6. **Configurable correlation method:** Allow the user to pass `method='spearman'` (default), `method='pearson'`, or `method='mutual_info'` (for nonlinear dependence — use `sklearn.feature_selection.mutual_info_regression` pairwise, normalized to [0,1]). Mutual info is slow — warn the user for >500 features.

**Key design decisions:**
- The clustering threshold is the main hyperparameter. Expose it clearly. Too low = every feature is its own cluster (reduces to standard importance). Too high = everything in one cluster (useless).
- Cache the correlation matrix computation — it's expensive for wide datasets.
- Support both a trained Booster object and a fitted LGBMRegressor/LGBMClassifier (extract the booster from `model.booster_`).

**Tests for Phase 3:**
- Create a dataset with 3 groups of known-correlated features (e.g., group A: 5 features that are just noisy copies of one signal, group B: same, group C: same). Train a model. Verify that the clustering correctly identifies the 3 groups.
- Verify that cluster-level importance sums to total importance.
- Verify that swapping which correlated feature within a group is "most important" doesn't change the cluster importance.
- Test stability analysis with 3 snapshots where feature importance is artificially stable for some clusters and shuffled for others.
- Test the `mutual_info` method on a small dataset.

---

### Phase 4: The Main Model — `IncrementalLGBM`

**File:** `drift_lgbm/model.py`

Build the `IncrementalLGBM` class that ties everything together. This is the user-facing API.

```python
class IncrementalLGBM:
    """
    Incremental LightGBM with distribution-aware rebinning,
    tree retirement, and correlation-aware feature importance.
    
    Scikit-learn compatible: implements fit(), partial_fit(), predict(), 
    predict_proba() (for classification), and get_params()/set_params().
    """
```

**Core API:**

```python
# First call: trains from scratch, establishes baseline bins and importance
model = IncrementalLGBM(task='regression', n_estimators_per_batch=100, **lgbm_params)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Subsequent calls: incremental updates with all the intelligence
model.partial_fit(X_new, y_new, eval_set=(X_val_new, y_val_new))

# Feature importance that accounts for correlation
report = model.feature_importance_report(X_val, y_val)
report.summary()

# Training history
model.history_  # list of dicts with metrics, drift info, trees retired per batch

# Predict
preds = model.predict(X_test)
```

**What `partial_fit` does internally (this is the important part):**

1. **Check distribution drift** using `DistributionTracker.update(X_new)`.
2. **If `should_rebin()` is True:**
   - Score all existing trees using `TreeManager.score_trees()`.
   - Retire negatively-contributing trees.
   - Retrain from scratch on the new batch (with the current hyperparameters), using the retired-tree model as `init_model` but with `keep_training_booster=False`. Actually no — if we need to rebin, we need a fresh booster. The strategy: take a *sample* of old data (if stored) combined with new data, train fresh, then port over the high-scoring old trees by appending them to the new model's text representation. If no old data is stored, just train fresh on the new batch. **Document this tradeoff clearly.**
   - Reset the distribution tracker with the new batch's distributions.
3. **If `should_rebin()` is False:**
   - Train new trees on the new batch using `init_model=existing_booster`.
   - Score all trees (including new ones) and retire any that now hurt performance.
4. **Update importance snapshot** for stability tracking.
5. **Log everything** to `history_`.

**Constructor parameters:**

```python
def __init__(
    self,
    task='regression',           # 'regression' or 'classification'
    n_estimators_per_batch=100,  # trees to add per partial_fit call
    max_total_trees=1000,        # hard cap on ensemble size
    drift_threshold=0.01,        # KS test p-value threshold
    drift_fraction=0.3,          # fraction of features drifted to trigger rebin
    retirement_threshold=0.0,    # tree contribution below this → retire
    correlation_method='spearman',
    cluster_distance_threshold=0.5,
    store_recent_data=True,      # keep last N batches for rebin retraining
    max_stored_batches=3,        # how many batches to keep
    verbose=1,
    **lgbm_params                # passed through to LightGBM
):
```

**Key design decisions:**
- `store_recent_data`: If True, keep the last N batches in memory so that when rebinning is triggered, we can retrain on recent data rather than just the latest batch. This costs memory but produces much better results. Default True with 3 batches.
- The `fit()` method MUST behave identically to a standard `LGBMRegressor.fit()` if you never call `partial_fit()`. A user should be able to swap this in with zero changes to their existing pipeline.
- Thread-safety is NOT a goal. Document this.
- All LightGBM parameters are passed through. We don't second-guess the user's choices on `num_leaves`, `learning_rate`, etc.

**Tests for Phase 4:**
- **Correctness baseline:** Train `IncrementalLGBM` with `fit()` only (no `partial_fit`). Compare predictions to a vanilla `LGBMRegressor` with the same params. They should be identical (or nearly so).
- **Incremental improvement:** Generate a synthetic dataset with 10,000 rows. Train on first 2,000. Then `partial_fit` with 2,000 more rows at a time. Final model should outperform the model trained on only the first 2,000.
- **Drift handling:** Generate data where features shift distribution halfway through. Verify that the model triggers rebinning and that post-rebin performance recovers.
- **Tree retirement:** Train incrementally 10 times. Verify that `max_total_trees` is never exceeded.
- **Feature importance report:** After incremental training, verify the report is produced and contains all expected fields.
- **Serialization:** pickle the model, unpickle it, verify predictions are identical.

---

### Phase 5: Synthetic Financial Data Generator (for testing)

**File:** `drift_lgbm/testing/synthetic.py`

Build a data generator that creates realistic-ish financial feature data for testing. This is crucial — all the above tests need data that actually exercises the financial use case.

```python
def generate_financial_data(
    n_samples=10000,
    n_informative=10,
    n_correlated_groups=3,       # groups of correlated features
    features_per_group=5,        # features in each correlated group
    n_noise=10,                  # pure noise features
    regime_change_at=None,       # sample index where distribution shifts
    regime_shift_magnitude=2.0,  # how much features shift at regime change
    target_noise=0.1,
    random_state=42
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns (X, y, metadata) where metadata describes the true 
    feature structure (which features are correlated, which are 
    informative, where the regime change is).
    """
```

**How it works:**
1. Create `n_correlated_groups` independent latent signals.
2. For each group, create `features_per_group` observable features as: `feature = signal + noise * epsilon`. Vary the noise level so some copies are cleaner than others.
3. Name features like real financial features: `momentum_5d`, `momentum_10d`, `momentum_21d`, `vol_realized_10d`, `vol_realized_21d`, `vol_implied_30d`, etc.
4. Target is a linear combination of the latent signals plus noise.
5. If `regime_change_at` is set, shift the mean/variance of the latent signals after that index.

**Tests:** Verify that the generated data actually has the correlation structure and regime change properties we specified.

---

### Phase 6: Integration Tests and Benchmarks

**File:** `tests/test_integration.py` and `benchmarks/benchmark_vs_vanilla.py`

**Integration tests:**
1. Full pipeline: generate synthetic data → fit → partial_fit 5 times → predict → get importance report. No crashes.
2. Regime change recovery: generate data with regime change. Train on pre-change data. Partial_fit on post-change data. Verify model adapts (performance on post-change test set improves over the pre-change model).
3. Correlation-aware importance correctness: generate data with known correlated groups. Verify clustered importance correctly identifies which groups are informative vs. noise.
4. Compare against naive `init_model`: run the same incremental procedure using raw LightGBM `init_model` without any of our system (no drift detection, no retirement, no rebinning). Compare performance on a held-out test set from the latest distribution. Our system should match or beat it, especially after distribution shifts.

**Benchmark:**
- Generate a dataset with 100,000 rows, 200 features, with a regime change at row 50,000.
- Method A: Train vanilla LightGBM on all 100,000 rows (the gold standard — full data access).
- Method B: Train vanilla LightGBM on first 50,000 only (the "can't fit all data" baseline).
- Method C: Raw `init_model` in 5 batches of 20,000.
- Method D: Our `IncrementalLGBM` in 5 batches of 20,000.
- Report RMSE and training time for each.
- **Expected outcome:** D ≥ C (we should be at least as good), B < C < A (more data helps), and D closer to A than C is to A (our system extracts more value from incremental training than naive init_model).

---

### Phase 7: Packaging and Documentation

**Files:** `pyproject.toml`, `README.md`, `drift_lgbm/__init__.py`

1. Package as `drift-lgbm` with proper dependencies: `lightgbm>=4.0`, `numpy`, `scipy`, `scikit-learn`, `pandas`.
2. `__init__.py` exports: `IncrementalLGBM`, `ClusteredImportanceReport`, `DistributionTracker`.
3. `README.md` with:
   - One-paragraph description of what this solves and why `init_model` isn't enough.
   - Installation instructions.
   - Quickstart code (5 lines: create, fit, partial_fit, predict, report).
   - "How it works" section explaining the three components.
   - "Prior work" section citing: Lopez de Prado's clustered feature importance, the Sgbt paper, the online GBDT paper, and LightGBM's own init_model.
   - License: MIT.

---

## DEFINITION OF DONE

The project is complete when:

1. All tests pass.
2. The benchmark runs and produces the expected ordering (D ≥ C).
3. The following 10-line script works end-to-end:

```python
from drift_lgbm import IncrementalLGBM

model = IncrementalLGBM(task='regression', n_estimators_per_batch=100)

# Initial training
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# New data arrives
model.partial_fit(X_new, y_new, eval_set=(X_val, y_val))

# What features matter now?
report = model.feature_importance_report(X_val, y_val)
report.summary()

# Predict
predictions = model.predict(X_test)
```

4. `pickle.dumps(model)` and `pickle.loads(...)` round-trips correctly.
5. README exists and is accurate.

---

## THINGS TO WATCH OUT FOR

1. **LightGBM model text format is undocumented and fragile.** The tree retirement module parses it. If this breaks, fall back to the alternative: instead of modifying the model text, keep a mask of "active" tree indices and zero out retired trees' predictions at inference time. Less clean but more robust.

2. **Memory management with `store_recent_data`.** The stored batches can get large. Use numpy arrays, not DataFrames, for storage. Consider storing only a random subsample (e.g., 10,000 rows per batch) if the batches are huge.

3. **The KS test can be noisy with small batches.** If a batch has <500 rows, increase the p-value threshold automatically (document this).

4. **Mutual information for clustering is O(n_features²).** For >500 features, default to Spearman with a warning, even if the user requested mutual_info.

5. **LightGBM's Python API has two interfaces: the sklearn API (LGBMRegressor) and the training API (lgb.train).** We use lgb.train internally because we need fine-grained control over the Booster. But our external API looks like sklearn.

6. **The model should never be WORSE than naive init_model.** If tree retirement or rebinning makes things worse on the validation set, skip that step and log a warning. Safety first.

---

## CITATION NOTES

If this gets published or open-sourced, cite:

- Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017. (The base model)
- Lopez de Prado, M. "Advances in Financial Machine Learning." Wiley 2018. (Clustered feature importance concept, Chapter 8)
- Lin, H. et al. "Online Gradient Boosting Decision Tree: In-Place Updates." arXiv 2502.01634, 2025. (Related work on online GBDT)
- Vaznais, M. et al. "Gradient Boosted Trees for Evolving Data Streams." Machine Learning, Springer 2024. (Streaming GBDT with drift detection)
- Chen, R. et al. "Deep Incremental Learning Models for Financial Temporal Tabular Datasets." arXiv 2303.07925, 2023. (Incremental learning specifically for quant finance)
