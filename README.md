# drift-lgbm

**`partial_fit` for LightGBM: incremental training on new data that detects drift, removes stale trees, and never does worse than plain `init_model`. Plus feature importance that holds up when features are correlated.**

LightGBM's `LGBMRegressor` and `LGBMClassifier` have no `partial_fit`. The usual way to update a LightGBM model with new data, `lgb.train(..., init_model=booster)`, only ever *adds* trees. It doesn't check whether your data drifted or whether the update helped, and trees fitted on an old market regime keep voting forever. `drift-lgbm` gives you an sklearn-style `IncrementalLGBM` with `fit()`, `partial_fit()` and `predict()` that runs those checks for you. It's built for financial time series and quant research, where regimes change and features (momentum, volatility, value, ...) are heavily correlated.

- **Incremental / online learning for LightGBM:** `partial_fit(X_new, y_new)` continues training on each new batch, like sklearn's `SGDRegressor.partial_fit`.
- **Data drift detection:** a per-feature KS test reports which features moved and by how much.
- **Stale-tree removal:** every tree is scored on fresh data, and the ones that now hurt are pruned (deleted) from the booster.
- **Safety check:** every update is compared against plain `init_model` and kept only if it is significantly better.
- **Correlation-aware feature importance:** correlated features are clustered (López de Prado's clustered feature importance), so importance and feature selection happen per cluster instead of splitting credit arbitrarily between near-duplicates.
- **Drop-in:** if you never call `partial_fit`, `fit()` gives bit-identical predictions to `LGBMRegressor.fit()`.

## Why `init_model` isn't enough

Continued training with `init_model` never removes anything. Trees fitted in a dead regime keep voting, and new trees spend capacity cancelling them out. If you build new batches' Datasets with `reference=` the original one (a common pattern, needed to continue a live training booster), the histogram bins also stay frozen at the first batch, and drifted values get crammed into edge bins: in our benchmark that costs **+34-57% test RMSE**. Meanwhile, standard gain importance (and SHAP) splits credit arbitrarily between correlated features (`momentum_10d` vs `momentum_21d`, five flavours of volatility), so dropping an "unimportant" feature just doubles its twin's importance. With cluster-level importance and stability tracking across batches, you can see which feature *clusters* matter this quarter versus last, and trust the answer.

## Installation

From a checkout of this repository (not yet on PyPI):

```bash
pip install .              # library
pip install -e ".[test]"   # development: editable install plus pytest
```

Requires Python ≥ 3.9 and `lightgbm>=4.0`, `numpy`, `scipy`, `scikit-learn`, `pandas`.

## Quickstart

```python
from drift_lgbm import IncrementalLGBM

model = IncrementalLGBM(task="regression", n_estimators_per_batch=100)
model.fit(X_train, y_train, eval_set=(X_val, y_val))       # identical to LGBMRegressor.fit
model.partial_fit(X_new, y_new, eval_set=(X_val, y_val))   # drift-aware incremental update
predictions = model.predict(X_test)
model.feature_importance_report(X_val, y_val).summary()    # what matters now, by cluster
```

A runnable version with synthetic data is in [`examples/quickstart.py`](examples/quickstart.py). Its output, abridged:

```
[drift-lgbm] fit: 100 trees on 10000 rows | val rmse 3.7123
[drift-lgbm] batch 1 | drift 71% (25 feat.) -> rebin_rejected | trees 100 -> 200 (retired 0, evicted 0, added 100) | prequential rmse 2.6559 | eval rmse 3.7123 -> 0.84712 | eval_set check: ours 0.8688 vs naive 0.84712
Clustered feature importance (method=spearman, distance_threshold=0.5, linkage=average, importance=gain)
35 features in 23 clusters; total gain 673,262.9; 2 snapshots, overall stability (Kendall tau) +0.66

  # cluster   share   importance stability    perm.rmse  members (intra-cluster share)
  1       1   22.5%    151,483.5     +0.82         2.33  vol_realized_21d (72%), vol_realized_10d (18%), vol_parkinson_21d (8%), vol_implied_30d (2%), +1 more
  2       2   16.7%    112,691.3     +0.91        1.389  value_div_yield (72%), value_pe (21%), value_pb (4%), value_ev_ebitda (2%), +1 more
  3       5   11.0%     73,727.7     +0.64       0.9645  skew_realized_63d (100%)
  ...
  8       0    6.2%     41,538.6     +0.91       0.8141  momentum_63d (56%), momentum_5d (24%), momentum_126d (8%), momentum_10d (6%), +1 more
  ...
 14      21    0.0%         77.3     +0.18    0.0001928  noise_08 (100%)
```

Note the intra-cluster shares: within the volatility family, `vol_realized_21d` takes 72% of the credit, but that split between near-substitutes is arbitrary. The cluster's 22.5% is the number to act on.

Every update is logged in `model.history_`: drift statistics, the action taken, trees retired, evicted and added, the prequential loss (how well the model predicted the batch *before* seeing it), and the safety-check comparison. `pd.DataFrame(model.history_)` gives a quick audit trail.

## Common questions

**Does LightGBM have `partial_fit`?**
No. As of LightGBM 4.x there is no LGBM partial fit: `LGBMRegressor` and `LGBMClassifier` only have `fit`. The built-in way to continue training a LightGBM model on new data is `lgb.train(params, new_data, init_model=old_booster)` (or `fit(..., init_model=...)`), which appends trees. `IncrementalLGBM.partial_fit` wraps that route and adds drift detection, tree pruning and a safety check.

**How do I update a LightGBM model with new data without retraining from scratch?**
Call `model.partial_fit(X_new, y_new)` each time a new batch arrives (daily, weekly, quarterly). Each call adds `n_estimators_per_batch` trees, removes trees that now hurt, and logs what it did to `model.history_`.

**Why did my LightGBM model get worse after continued training?**
There are two common causes. (1) Frozen bins: the new data's Dataset was built with `reference=` the first one, so values outside the old range fall into edge bins. (2) Dead-regime trees: trees fitted on old data keep voting after the relationship changed. `IncrementalLGBM` always builds fresh Datasets, and it retires trees that measurably hurt on new data.

**How do I remove or delete trees from a LightGBM Booster?**
LightGBM has no public API for removing arbitrary trees. `TreeManager` does it, and it can also tell you *which* trees to remove:

```python
from drift_lgbm import TreeManager

tm = TreeManager(booster)
ranking = tm.score_trees(X_val, y_val)         # (tree_index, contribution), most helpful first
harmful = tm.identify_retiring(threshold=0.0)  # trees whose removal improves validation loss
smaller = tm.retire_trees(booster, harmful)    # a new Booster; the input is untouched
```

A single call never removes more than 50% of the trees.

**How do I detect data drift in my features?**
`DistributionTracker` works on its own with any DataFrame, no model needed:

```python
from drift_lgbm import DistributionTracker

tracker = DistributionTracker()
tracker.update(X_last_quarter)            # sets the reference distribution
report = tracker.update(X_this_quarter)
report.summary()                          # drifted features, KS statistic, p-value, mean shift
```

**Why is feature importance (or SHAP) misleading with correlated features?**
When two features carry the same information, the model uses them interchangeably, so the split of credit between them is noise. Drop one, and the other's importance roughly doubles. `ClusteredImportance` groups correlated features and reports importance per cluster. It works on any LightGBM model, including a plain `LGBMRegressor`:

```python
from drift_lgbm import ClusteredImportance

ClusteredImportance(distance_threshold=0.5).compute(model, X_val, y_val).summary()
```

It aggregates LightGBM gain importance and adds a joint permutation importance per cluster. It does not compute SHAP values.

**Can I use it for walk-forward / rolling-window retraining?**
Partly. `partial_fit` forgets by *performance*, not by age: an old tree is removed only when it measurably hurts on new data, so a strict "always drop the oldest quarter" window is not built in. For a strict rolling window, call `fit()` on each window. `DistributionTracker` and `ClusteredImportance` still work standalone for drift and importance across windows.

**Is it a drop-in replacement for `LGBMRegressor`?**
For `fit` and `predict`, yes. `fit()` delegates to `LGBMRegressor`/`LGBMClassifier`, so predictions are bit-identical until you call `partial_fit`. It also supports `get_params`/`set_params`/`clone` and pickling.

## How it works

### 1. `DistributionTracker`: when has the data moved?
- Keeps its **own** per-feature histograms (256 quantile bins from the reference batch), not LightGBM's.
- Each new batch is KS-tested against the reference. To avoid storing raw data, the test runs on the quantile grid: it compares the empirical CDFs at the bin edges and their left limits. That statistic can only under-estimate the exact KS distance, so p-values are conservative. A feature is *drifted* if `p < drift_threshold` (0.01).
- `should_rebin()` is True when **more than 30%** of features drifted.
- Batches under 500 rows get a proportionally looser threshold (capped at 0.1), because the KS test has little power on small samples.
- Reports per feature the KS distance, p-value, standardized mean shift, std ratio, and **out-of-range fraction**: the mass that first-batch bins cannot resolve.
- NaN and ±inf are ignored and counted separately. The tracker is picklable.

### 2. `TreeManager`: which trees still help?
- A boosted ensemble is additive in raw-score space, and LightGBM folds the base score and the learning rate into the leaf values. So removing tree *j* gives exactly `F(x) − tree_j(x)`. All leave-one-out contributions come from one `pred_leaf` call plus the leaf values parsed from the model text: 1000 trees × 5000 rows are scored in about 0.3 s.
- `score_trees(X_val, y_val, metric)` supports rmse, mae, logloss, auc or a callable. `identify_retiring(threshold)` returns the trees below the threshold. `select_retirement` confirms removals greedily and only keeps those that improve the loss **significantly** (two standard errors), so validation noise does not retire neutral trees.
- `retire_trees(booster, indices)` returns a **new** Booster. It never removes more than 50% of the trees in one call. **Implementation note (a deliberate hack):** it cuts `Tree=` blocks out of `model_to_string()`, renumbers the rest, recomputes the `tree_sizes` header (LightGBM hard-crashes on a wrong one), and reloads the text. The format is undocumented; this is verified on LightGBM 4.x and pinned by the tests.

### 3. `ClusteredImportance`: importance that survives correlation
- Dependence is measured by Spearman (the default), Pearson, or pairwise mutual information. MI is mapped to [0, 1] with Linfoot's `sqrt(1 − e^(−2·MI))`, which equals |ρ| for Gaussians, so the same threshold means the same thing. It falls back to Spearman above 500 features.
- `distance = 1 − |dependence|`, then agglomerative clustering (average linkage) cut at `distance_threshold` (0.5). This is the main knob: 0 gives plain per-feature importance, 1 puts everything in one cluster.
- **Cluster importance** is the sum of member gain, so the cluster totals add up to the model total. Intra-cluster shares are informational only. Select and drop features at the cluster level.
- **Stability**: per-batch importance snapshots are scored by Kendall's tau, decomposed per cluster: does the cluster keep its rank relative to the others from batch to batch? The scores average to the global tau.
- With `y`, the report also includes **clustered permutation importance**: each cluster's columns are permuted jointly on validation data, so correlated substitutes cannot cover for each other.
- Accepts a `Booster`, a fitted `LGBMRegressor`/`LGBMClassifier`, or an `IncrementalLGBM`. Dependence matrices are cached by content hash.

### 4. `IncrementalLGBM.partial_fit`: putting it together
1. **Drift check** against the tracker's reference.
2. **Score and prune** the existing trees on data they have not seen: the `eval_set`, or else the new batch itself. Retire significantly harmful trees and evict the weakest if `max_total_trees` needs room. This happens *before* training: new trees fitted on top of a stale tree partly compensate for it, which would then make it look useful.
3. **Train** `n_estimators_per_batch` trees with the pruned ensemble as `init_model`. On drift (a **rebin**), the fresh `lgb.Dataset`, and hence fresh bins, is built from the new batch plus any *stored recent batches that still match its distribution*. Batches from the dead regime fail the same KS rule and are left out. The surviving old trees are "ported" as `init_model`, so the fresh trees learn only the residual.
4. **Post-training pruning** with an `eval_set`.
5. **Safety check**: if anything differs from a plain `init_model` update, that naive candidate is trained too. Ours is kept only if it is *significantly* better (a one-sided 95% test on per-row losses); otherwise the naive update is used and logged (`rebin_rejected` / `retirement_rejected`). Without an `eval_set`, candidates are compared on the most recent 20% of the batch (walk-forward), and the winning strategy is **refit on the whole batch**, so no training data is lost.
6. Reset the tracker after a rebin, store the batch (numpy, subsampled to 10k rows), snapshot importance, and log to `history_`.

`fit()` delegates to LightGBM's own `LGBMRegressor`/`LGBMClassifier`, so if you never call `partial_fit` it is a zero-change drop-in with identical predictions (tested bit-for-bit, including early stopping).

## Benchmark

`python benchmarks/benchmark_vs_vanilla.py` runs 100,000 rows × 200 features (10 correlated factor families × 10, 40 stand-alone signals, 60 noise), with a regime change at row 50,000 and 5 batches of 20,000. The test set is 20,000 rows from the post-change ("latest") distribution. All methods use the same LightGBM parameters (lr 0.1, 31 leaves), and A/B train 500 trees, the same capacity as C/D. There are two scenarios. **Covariate**: factor means and volatilities shift, but y|x does not change. **Concept**: the same, plus half the factor premia flip sign.

Seed 0 (test RMSE, % vs naive `init_model`):

| Method | Covariate shift | Covariate + concept drift |
|---|---:|---:|
| A: vanilla LightGBM, all 100k rows | 1.8604 (+6.0%) | 2.0067 (−12.0%) |
| B: vanilla LightGBM, first 50k rows | 8.7730 (+399.8%) | 9.0996 (+299.0%) |
| C: naive `init_model`, 5 × 20k | 1.7553 | 2.2808 |
| C*: naive `init_model`, **frozen bins** (`reference=`) | 2.7472 (+56.5%) | 3.0581 (+34.1%) |
| **D: `IncrementalLGBM`, 5 × 20k** | **1.7553 (±0.0%)** | **2.1461 (−5.9%)** |
| training time C → D | 2.9 s → 8.8 s | 3.0 s → 12.3 s |

D/C RMSE ratio across seeds 0, 1, 2: covariate **1.000, 1.000, 1.000**; concept **0.941, 0.988, 0.926**. D ≥ C in all six runs.

What this shows, honestly:
- **D ≥ C holds in every run.** Under pure covariate shift the old trees are still correct (y|x is unchanged). The safety check correctly finds that neither pruning nor rebuilding is significantly better, and D reproduces the naive update exactly. Under concept drift D retires dead-regime trees at the change (52-102 trees per run) and beats naive `init_model` by 1-7%.
- **Frozen bins are the real `init_model` trap.** Building each batch's Dataset with `reference=` the first batch (C*) is 34-57% worse. D always builds fresh Datasets. Note that a plain `lgb.train(..., init_model=...)` on a *fresh* Dataset (C) already re-derives bins from the new batch, so the "frozen bins" problem is specific to the `reference=` / continued-booster pattern.
- **The intuitive "more data helps" ordering (B < C < A) does not always hold.** Under covariate shift, naive incremental training (C) *beats* retraining on all 100k rows (A) on the post-change test set, because A spends capacity on the old regime while C's later trees specialise on the new one. Under concept drift, A is best, and D closes 9-61% of the gap between C and A (seeds 0-2).
- D costs about 3-4× C's training time: it trains extra candidate models and scores trees.

Reproduce with `python benchmarks/benchmark_vs_vanilla.py [--seeds 0 1 2] [--quick] [--output results.json]`. The exit status is 0 iff D ≥ C in every run.

## Design decisions

Each of these replaced a more obvious approach after measuring it. Each is documented at the relevant place in the code.

| Obvious approach | What we do | Why |
|---|---|---|
| Rebin: train fresh, then *append* high-scoring old trees to the model text | Surviving old trees become the `init_model` of a fresh Dataset | Two independently trained full predictors, concatenated, double-count the target |
| Rebin trains on all stored recent batches | Only stored batches whose distribution still matches the new batch | At a regime change, the stored batches *are* the dead regime (and already fitted by the ported trees); using them was measurably worse than naive |
| Score/retire trees after training | Prune *before* training (plus after, with an `eval_set`) | New trees compensate for stale ones, masking them in LOO scores |
| Retire if contribution < threshold | …and the removal must be significant (2 standard errors) | Otherwise ~40% of neutral trees are retired by validation noise |
| "Skip a step if it makes validation worse" | Keep our step only if significantly better (one-sided 95%) | Ties go to the naive update; this removed selection-noise losses under covariate shift |
| — | Without `eval_set`: select on the latest 20% of the batch, then refit on 100% | A plain hold-out cost ~5% RMSE in lost training data |
| KS test via `scipy.stats.kstwo` | Asymptotic Kolmogorov distribution with Stephens' correction | `kstwo.sf` took ~7 s per 200-feature batch; this takes ~0 s and is within ~10% for n ≥ 25 |
| "Rank correlation per cluster" | Per-cluster decomposition of Kendall's tau | Gives a per-cluster score that averages to the global tau |
| `fit()` must match `LGBMRegressor.fit()` | `fit()` *delegates* to `LGBMRegressor`/`LGBMClassifier` | Guarantees identical parameter handling; `partial_fit` reuses the exact processed params |
| (extra) | `generate_financial_data(..., regime_concept_shift=)` | A regime change that only shifts features is covariate shift, where old trees never "die"; concept drift is the case the system is built for |

## Configuration reference

```python
IncrementalLGBM(
    task="regression",              # or "classification" (binary)
    n_estimators_per_batch=100,     # trees per fit / partial_fit (n_estimators= is accepted as an alias)
    max_total_trees=1000,           # hard cap; weakest trees are evicted first
    drift_threshold=0.01,           # KS p-value threshold
    drift_fraction=0.3,             # > 30% of features drifted -> rebin
    retirement_threshold=0.0,       # LOO contribution below this -> retirement candidate
    correlation_method="spearman",  # or "pearson", "mutual_info"
    cluster_distance_threshold=0.5,
    store_recent_data=True,         # keep recent batches for rebins
    max_stored_batches=3,
    verbose=1,                      # drift-lgbm's logging; LightGBM's is `verbosity` (silent by default)
    max_rows_per_stored_batch=10_000,
    validation_fraction=0.2,        # walk-forward selection share when no eval_set
    safety_checks=True,             # compare against the naive init_model update
    scoring_metric=None,            # rmse / logloss by default; mae, auc or a callable
    **lgbm_params,                  # passed through unchanged (learning_rate, num_leaves, random_state, ...)
)
```

The model is sklearn-compatible (`get_params`/`set_params`/`clone`, `score`, `predict_proba` for classification) and picklable, including the full training state.

## Limitations
- Regression and **binary** classification only. Tree scoring needs single-output models with a built-in objective, so there is no multiclass, ranking, random-forest mode, or custom objective callables.
- Numeric features only: encode categoricals yourself.
- `partial_fit` trains exactly `n_estimators_per_batch` trees; early stopping applies to `fit` only.
- Retirement and the safety check are *selected* on validation data, so post-update validation metrics are optimistic. Judge the model on a separate test set.
- **Not thread-safe.** Do not call `partial_fit` on one instance from several threads.
- Tree surgery relies on LightGBM's undocumented model-text format (tested on 4.x). If it ever breaks, the documented fallback is to keep a mask of active trees and subtract retired trees at predict time.

## Development

```bash
pip install -e ".[test]"
pytest                                            # full suite, ~20 s
python benchmarks/benchmark_vs_vanilla.py --quick # ~10 s benchmark
```

## Prior work

- Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. The base model, and the `init_model` continued-training mechanism this library builds on.
- López de Prado, M. *Advances in Financial Machine Learning.* Wiley, 2018, ch. 8: MDI/MDA feature importance and the substitution effect among correlated features.
- López de Prado, M. *Machine Learning for Asset Managers.* Cambridge University Press, 2020: clustered feature importance (clustered MDI/MDA), which our cluster-level importance and joint-permutation importance follow.
- Gunasekara, N., Pfahringer, B., Gomes, H. M., Bifet, A. *Gradient boosted trees for evolving data streams* (SGBT). Machine Learning 113, 3325-3352, 2024. Streaming GBDT with drift detection and tree replacement.
- Lin, H., Chung, J. W., Lao, Y., Zhao, W. *Online Gradient Boosting Decision Tree: In-Place Updates for Efficient Adding/Deleting Data.* arXiv:2502.01634, 2025.
- Wong, T., Barahona, M. *Deep incremental learning models for financial temporal tabular datasets with distribution shifts.* arXiv:2303.07925, 2023.

## License

MIT. See [LICENSE](LICENSE).
