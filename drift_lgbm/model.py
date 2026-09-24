"""``IncrementalLGBM``: the user-facing incremental model.

It ties the three components together:

* :class:`~drift_lgbm.distribution_tracker.DistributionTracker` decides
  *when* the feature distributions have moved enough to warrant a rebuild.
* :class:`~drift_lgbm.tree_manager.TreeManager` scores every tree on fresh
  data and retires trees that now hurt.
* :class:`~drift_lgbm.feature_importance.ClusteredImportance` reports which
  feature *clusters* matter, and how stably, across batches.

What ``partial_fit`` does
-------------------------
1. **Drift check**: KS-test the new batch against the tracker's reference.
2. **Score and prune**: score every existing tree (leave-one-out) on data it
   has not seen: the ``eval_set`` if given, otherwise the new batch itself
   (prequential). Retire trees that hurt, keeping only removals that improve
   the loss *significantly* (2 standard errors), and evict the weakest trees
   if the ``max_total_trees`` cap needs room. All removals in one call are
   capped at 50% of the ensemble. Pruning happens *before* training on
   purpose. Pruning afterwards would be much less effective: new trees are
   fitted to the residuals of the old ones, so they partly "compensate" stale
   trees, and a stale tree then looks useful in leave-one-out scoring.
3. **Train**:

   * *No drift*: add ``n_estimators_per_batch`` trees on the new batch, with
     the pruned ensemble as ``init_model``.
   * *Drift ("rebin")*: build a fresh ``lgb.Dataset``, and hence fresh
     histogram bins, from the new batch plus those stored recent batches that
     still match its distribution. Stored batches from the dead regime fail
     the same KS drift rule and are left out. Train on it with the
     *surviving, still-useful* old trees as ``init_model``, so they are
     "ported" into the new model. **Tradeoff**: a from-scratch model on
     recent data alone would discard everything the old trees still know,
     while keeping every old tree lets dead regimes keep voting. We keep the
     trees that the scoring data says still help, and the fresh trees learn
     only the residual. (Training a fresh model independently and appending
     old trees to its text would double-count the target, since both would
     be full predictors. Using the ported trees as ``init_model`` is the
     consistent version.) The tracker is then reset to the new batch's
     distribution.

4. **Post-training pruning** (only with an ``eval_set``, the only data the
   new trees have not seen): score all trees again, the new ones included,
   and retire any that hurt.
5. **Safety check**: the model must never end up worse than naive
   ``init_model``. Whenever steps 2-4 changed the update relative to a plain
   continuation (a rebin, or pruning), a naive candidate is also trained. Our
   update is kept only if it is *significantly* better on validation data (a
   one-sided 95% test on the per-row loss difference); otherwise the naive
   update is used, and the step is logged in ``history_`` as
   ``rebin_rejected`` or ``retirement_rejected``. If our update was actually
   worse, a :class:`DriftLGBMWarning` is emitted as well.
6. **Book-keeping**: store the batch for future rebins, snapshot the new trees'
   feature importance for stability analysis, and log everything to
   ``history_``, including the *prequential* loss: how well the model
   predicted the batch before seeing it.

A note on "frozen bins"
-----------------------
Every call to ``lgb.train(..., init_model=...)`` here builds a *new*
``lgb.Dataset``, so LightGBM derives new histogram bins from the data it is
given. Bins are frozen at the first batch only when a new Dataset is built with
``reference=`` the original one, or when training continues on the same
training booster. That is the setup where drifted values get crammed into
edge bins. The tracker's ``out_of_range_fraction`` measures exactly that mass.
In this library, "rebin" means rebuilding on the current regime's recent data,
so the bins, and the trees, reflect the regime the model now operates in.

Validation data
---------------
With ``eval_set=(X_val, y_val)``, all scoring and the safety check use it,
and the model trains on the whole batch. Without one, existing trees are
scored on the new batch, which they have not seen. When the safety check has
a decision to make, both candidates are trained on the older
``1 - validation_fraction`` of the batch and compared on its most recent rows
(rows are assumed to be in time order, so this is a walk-forward split). The
winning *strategy* is then refit on the whole batch, so no training data is
lost. These decisions are *selected* on the validation data, so validation
metrics after an update are optimistically biased. Assess final performance
on a separate test set.

Thread safety is **not** a goal: do not call ``partial_fit`` on the same
instance from several threads.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import deque
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metaestimators import available_if

from ._utils import align_columns, as_1d, as_2d_float
from .distribution_tracker import DistributionTracker, DriftReport
from .feature_importance import ClusteredImportance, ClusteredImportanceReport
from .tree_manager import TreeManager, booster_objective, loss_function, per_tree_importance, row_loss_function

__all__ = ["IncrementalLGBM", "DriftLGBMWarning"]

# Parameter aliases LightGBM uses for the number of boosting rounds and for
# early stopping (from lightgbm.basic._ConfigAliases; copied to avoid a
# private import). partial_fit always trains exactly the requested number of trees.
_NUM_ITERATION_ALIASES = frozenset(
    {
        "num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round",
        "num_rounds", "nrounds", "num_boost_round", "n_estimators", "max_iter",
    }
)
_EARLY_STOPPING_ALIASES = frozenset(
    {"early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change"}
)
# Held-out validation needs at least this many rows (and leaves at least this
# many for training); otherwise partial_fit falls back to prequential scoring.
_MIN_VALIDATION_ROWS = 20
# Significance gate for retiring a tree (see TreeManager.select_retirement).
# Two standard errors keeps validation noise from retiring neutral trees.
_RETIREMENT_MIN_Z = 2.0
# Safety check: our update replaces the plain init_model update only if it is
# better on validation data by this many standard errors of the per-row loss
# difference (a one-sided test at 95%). Ties and noise go to the naive update
# ("safety first"). On the synthetic benchmarks, a looser z = 1 let
# selection noise make pure-covariate-shift runs ~1% worse than naive, while
# z = 2 threw away real gains under concept drift.
_SAFETY_MIN_Z = 1.645


class DriftLGBMWarning(UserWarning):
    """Emitted when a safety check overrides a drift-lgbm step (e.g. a rebin is rejected)."""


def _normalize_eval_set(eval_set) -> Optional[List[Tuple[Any, Any]]]:
    """Accept ``(X, y)`` or ``[(X, y), ...]`` and return a list of pairs."""
    if eval_set is None:
        return None
    if isinstance(eval_set, tuple) and len(eval_set) == 2 and not isinstance(eval_set[0], tuple):
        return [eval_set]
    pairs = list(eval_set)
    if not pairs or not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in pairs):
        raise ValueError("eval_set must be (X_val, y_val) or a list of such pairs.")
    return [tuple(p) for p in pairs]


class IncrementalLGBM(BaseEstimator):
    """Incremental LightGBM with distribution-aware rebinning, tree retirement,
    and correlation-aware feature importance.

    Scikit-learn compatible: implements ``fit()``, ``partial_fit()``,
    ``predict()``, ``predict_proba()`` (for classification), ``score()``, and
    ``get_params()``/``set_params()``. If ``partial_fit`` is never called,
    ``fit`` is exactly ``LGBMRegressor.fit`` / ``LGBMClassifier.fit``, and
    predictions are identical.

    Parameters
    ----------
    task : {'regression', 'classification'}, default='regression'
        Classification is binary only.
    n_estimators_per_batch : int, default=100
        Trees trained by ``fit`` and added by each ``partial_fit``. If
        ``n_estimators`` is passed in ``**lgbm_params`` it takes precedence,
        which keeps ``IncrementalLGBM(n_estimators=...)`` a drop-in for
        ``LGBMRegressor``.
    max_total_trees : int, default=1000
        Hard cap on ensemble size. When a batch would exceed it, the
        lowest-scoring trees are evicted first. If the 50% per-call retirement
        guardrail forbids evicting enough, fewer new trees are trained.
    drift_threshold : float, default=0.01
        KS-test p-value below which a feature counts as drifted.
    drift_fraction : float, default=0.3
        A rebin is triggered when strictly more than this fraction of features
        drifted.
    retirement_threshold : float, default=0.0
        Trees with a leave-one-out contribution below this (0 = trees that hurt)
        are candidates for retirement.
    correlation_method : {'spearman', 'pearson', 'mutual_info'}, default='spearman'
        Dependence measure for :meth:`feature_importance_report`.
    cluster_distance_threshold : float, default=0.5
        Clustering cut for :meth:`feature_importance_report`
        (``distance = 1 - |correlation|``).
    store_recent_data : bool, default=True
        Keep the last ``max_stored_batches`` batches in memory so a rebin can
        train on a window of recent data rather than one batch. Only stored
        batches whose distribution still matches the new batch are used.
    max_stored_batches : int, default=3
        Number of past batches kept for rebins.
    verbose : int, default=1
        ``>= 1`` prints one line per fit / partial_fit. This is drift-lgbm's
        own logging. Use ``verbosity=`` to control LightGBM's, which defaults
        to silent here and does not change the model.
    max_rows_per_stored_batch : int, default=10_000
        Stored batches larger than this are randomly subsampled, which bounds
        memory. Storage uses numpy arrays.
    validation_fraction : float, default=0.2
        When ``partial_fit`` gets no ``eval_set`` and the safety check has a
        decision to make, this trailing fraction of the batch (the most recent
        rows) is used to choose between candidates. The winner is then refit on
        the whole batch. 0 disables the check when there is no ``eval_set``.
    safety_checks : bool, default=True
        Train a naive ``init_model`` candidate whenever a rebin or pruning
        changes the update, and fall back to it if it validates better.
        False skips that extra training (about 2-3x cheaper on those batches).
    scoring_metric : {'rmse', 'mae', 'logloss', 'auc'} or callable, optional
        Metric for tree scoring and safety checks. Defaults to rmse for
        regression and logloss for classification.
    **lgbm_params
        Passed through to LightGBM unchanged, in LGBMRegressor / LGBMClassifier
        style (``learning_rate``, ``num_leaves``, ``random_state``,
        ``class_weight``, ...).

    Attributes
    ----------
    booster_ : lightgbm.Booster
        The current ensemble.
    history_ : list of dict
        One record per ``fit`` / ``partial_fit`` call: drift statistics, the
        action taken (``initial_fit``, ``incremental``, ``rebin``,
        ``rebin_rejected`` or ``retirement_rejected``), trees
        retired/evicted/added, validation losses, timing and notes.
        ``val_loss_*`` values are losses (lower is better): the metric itself
        for rmse/mae/logloss, and ``1 - AUC`` for AUC. ``val_loss_before`` is
        the model *before* the update, scored on the new validation data.
    drift_reports_ : list of DriftReport
        Full drift report of every batch.
    importance_snapshots_ : list of dict
        Gain importance of the trees *added* in each batch
        (feature -> gain), used for stability analysis.
    tracker_ : DistributionTracker
    classes_ : ndarray (classification only)
    evals_result_ : dict
        Evaluation history from ``fit`` (LightGBM format).
    """

    def __init__(
        self,
        task: str = "regression",
        n_estimators_per_batch: int = 100,
        max_total_trees: int = 1000,
        drift_threshold: float = 0.01,
        drift_fraction: float = 0.3,
        retirement_threshold: float = 0.0,
        correlation_method: str = "spearman",
        cluster_distance_threshold: float = 0.5,
        store_recent_data: bool = True,
        max_stored_batches: int = 3,
        verbose: int = 1,
        max_rows_per_stored_batch: int = 10_000,
        validation_fraction: float = 0.2,
        safety_checks: bool = True,
        scoring_metric=None,
        **lgbm_params,
    ):
        self.task = task
        self.n_estimators_per_batch = n_estimators_per_batch
        self.max_total_trees = max_total_trees
        self.drift_threshold = drift_threshold
        self.drift_fraction = drift_fraction
        self.retirement_threshold = retirement_threshold
        self.correlation_method = correlation_method
        self.cluster_distance_threshold = cluster_distance_threshold
        self.store_recent_data = store_recent_data
        self.max_stored_batches = max_stored_batches
        self.verbose = verbose
        self.max_rows_per_stored_batch = max_rows_per_stored_batch
        self.validation_fraction = validation_fraction
        self.safety_checks = safety_checks
        self.scoring_metric = scoring_metric
        self._lgbm_params: Dict[str, Any] = dict(lgbm_params)

    # ================================================================== sklearn plumbing

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(self._lgbm_params)
        return params

    def set_params(self, **params) -> "IncrementalLGBM":
        own = set(self._get_param_names())
        for key, value in params.items():
            if key in own:
                setattr(self, key, value)
            else:
                self._lgbm_params[key] = value
        return self

    @property
    def _estimator_type(self) -> str:
        return "classifier" if self.task == "classification" else "regressor"

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "booster_")

    def _check_fitted(self) -> None:
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("This IncrementalLGBM instance is not fitted yet; call fit() first.")

    @property
    def n_trees_(self) -> int:
        self._check_fitted()
        return self.booster_.num_trees()

    @property
    def feature_importances_(self) -> np.ndarray:
        """Per-feature importance of the current ensemble (``importance_type`` as in LightGBM's wrapper)."""
        self._check_fitted()
        return self.booster_.feature_importance(importance_type=self._lgbm_params.get("importance_type", "split"))

    @property
    def drift_report_(self) -> DriftReport:
        """Drift report of the most recent batch."""
        self._check_fitted()
        return self.drift_reports_[-1]

    # ================================================================== validation helpers

    def _batch_size(self) -> int:
        return int(self._lgbm_params.get("n_estimators", self.n_estimators_per_batch))

    def _validate_params(self) -> None:
        if self.task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {self.task!r}.")
        n_batch = self._batch_size()
        if n_batch < 1:
            raise ValueError("n_estimators_per_batch must be >= 1.")
        if self.max_total_trees < n_batch:
            raise ValueError(
                f"max_total_trees ({self.max_total_trees}) must be >= n_estimators_per_batch ({n_batch})."
            )
        if not 0.0 < self.drift_threshold < 1.0:
            raise ValueError("drift_threshold must be in (0, 1).")
        if not 0.0 <= self.drift_fraction < 1.0:
            raise ValueError("drift_fraction must be in [0, 1).")
        if self.store_recent_data and self.max_stored_batches < 1:
            raise ValueError("max_stored_batches must be >= 1 when store_recent_data=True.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.correlation_method not in ("spearman", "pearson", "mutual_info"):
            raise ValueError("correlation_method must be 'spearman', 'pearson' or 'mutual_info'.")
        if not 0.0 <= self.cluster_distance_threshold <= 1.0:
            raise ValueError("cluster_distance_threshold must be in [0, 1].")
        rounds = (_NUM_ITERATION_ALIASES - {"n_estimators"}) & set(self._lgbm_params)
        if rounds:
            raise ValueError(f"Use n_estimators_per_batch instead of {sorted(rounds)}.")
        objective = self._lgbm_params.get("objective")
        if callable(objective):
            raise NotImplementedError("Custom objective callables are not supported (tree scoring needs a built-in objective).")
        if objective is not None and self.task == "classification" and objective not in (
            "binary", "cross_entropy", "xentropy"
        ):
            raise ValueError(f"objective={objective!r} does not match task='classification' (binary only).")
        if objective in ("binary", "multiclass", "multiclassova", "lambdarank") and self.task == "regression":
            raise ValueError(f"objective={objective!r} does not match task='regression'.")

    def _make_sklearn_estimator(self):
        kwargs = {k: v for k, v in self._lgbm_params.items() if k != "n_estimators"}
        # LightGBM's own logging is silenced unless asked for; it never changes the model.
        kwargs.setdefault("verbosity", -1)
        cls = lgb.LGBMClassifier if self.task == "classification" else lgb.LGBMRegressor
        return cls(n_estimators=self._batch_size(), **kwargs)

    def _prepare_X(self, X) -> np.ndarray:
        arr, names = as_2d_float(X)
        return align_columns(arr, names, self._report_names, self._input_names is not None)

    def _encode_y(self, y) -> np.ndarray:
        y = as_1d(y)
        if self.task == "regression":
            return y.astype(np.float64)
        idx = np.searchsorted(self.classes_, y)
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        unseen = self.classes_[idx] != y
        if np.any(unseen):
            raise ValueError(f"y contains labels not seen during fit: {np.unique(y[unseen])[:5]}")
        return idx.astype(np.float64)

    def _training_weights(self, y: np.ndarray, w: Optional[np.ndarray]) -> Optional[np.ndarray]:
        class_weight = self._lgbm_params.get("class_weight")
        if self.task != "classification" or class_weight is None:
            return w
        if isinstance(class_weight, dict):
            # Keys are original labels; y here is encoded as 0/1.
            class_weight = {i: class_weight[c] for i, c in enumerate(self.classes_) if c in class_weight}
        cw = compute_sample_weight(class_weight, y)
        return cw if w is None else w * cw

    def _loss_fn(self):
        return loss_function(self._objective, self._objective_options, self.scoring_metric)

    def _naive_wins(self, y: np.ndarray, raw_ours: np.ndarray, raw_naive: np.ndarray) -> Tuple[bool, float, float]:
        """Safety check. Returns ``(use_naive, loss_ours, loss_naive)``.

        Ours must beat the plain init_model update by ``_SAFETY_MIN_Z``
        standard errors of the mean per-row loss difference. Otherwise the
        plain update is kept, because under noise the conservative choice is
        the naive one. Metrics without per-row losses (AUC, callables) use a
        plain comparison.
        """
        _, loss = self._loss_fn()
        loss_ours, loss_naive = loss(y, raw_ours), loss(y, raw_naive)
        row_loss = row_loss_function(self._objective, self._objective_options, self.scoring_metric)
        if row_loss is None or len(y) < 2:
            return loss_ours > loss_naive, loss_ours, loss_naive
        diff = row_loss(y, raw_ours) - row_loss(y, raw_naive)
        std_err = float(np.std(diff, ddof=1)) / math.sqrt(len(diff))
        ours_better = diff.mean() < 0 and diff.mean() < -_SAFETY_MIN_Z * std_err
        return not ours_better, loss_ours, loss_naive

    def _raw(self, booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
        if booster.num_trees() == 0:
            return np.zeros(X.shape[0])
        return booster.predict(X, raw_score=True, num_iteration=-1)

    def _log(self, message: str) -> None:
        if self.verbose and self.verbose > 0:
            print(f"[drift-lgbm] {message}")

    # ================================================================== fit

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
    ) -> "IncrementalLGBM":
        """Train from scratch, exactly like ``LGBMRegressor.fit`` / ``LGBMClassifier.fit``.

        Training is delegated to LightGBM's own sklearn estimator, which
        guarantees identical parameter handling, label encoding and
        predictions. This also establishes the drift tracker's reference
        distribution, the first importance snapshot and ``history_[0]``.
        ``eval_set`` may be ``(X_val, y_val)`` or a list of pairs.
        """
        start = time.perf_counter()
        self._validate_params()
        X_arr, input_names = as_2d_float(X)
        y_arr = as_1d(y)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]}.")
        if self.task == "classification" and len(np.unique(y_arr)) > 2:
            raise NotImplementedError("Only binary classification is supported.")
        eval_list = _normalize_eval_set(eval_set)

        est = self._make_sklearn_estimator()
        est.fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_list,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
        )
        booster = est.booster_
        best = booster.best_iteration
        if best and 0 < best < booster.current_iteration():
            # Early stopping: LightGBM predicts with best_iteration by default.
            # Trimming keeps predictions identical and gives partial_fit the best model.
            booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))

        excluded = _NUM_ITERATION_ALIASES | _EARLY_STOPPING_ALIASES
        self._train_params = {k: v for k, v in est.booster_.params.items() if k not in excluded}
        self.booster_ = booster
        self._objective, self._objective_options = booster_objective(booster)
        if self.task == "classification":
            self.classes_ = np.asarray(est.classes_)
            self.n_classes_ = len(self.classes_)
        self.evals_result_ = getattr(est, "evals_result_", {}) or {}
        self.feature_name_ = booster.feature_name()
        self._input_names = list(input_names) if input_names is not None else None
        self._report_names: List[Hashable] = list(input_names) if input_names is not None else list(self.feature_name_)
        self.n_features_in_ = X_arr.shape[1]
        if input_names is not None:
            self.feature_names_in_ = np.asarray([str(n) for n in input_names], dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self._categorical_feature = categorical_feature
        seed = self._train_params.get("random_state")
        self._rng = np.random.default_rng(seed if isinstance(seed, (int, np.integer)) else None)

        self.tracker_ = DistributionTracker(drift_threshold=self.drift_threshold, drift_fraction=self.drift_fraction)
        self.drift_reports_ = [self.tracker_.update(pd.DataFrame(X_arr, columns=self._report_names))]

        y_enc = self._encode_y(y_arr)
        w_arr = None if sample_weight is None else as_1d(sample_weight, "sample_weight").astype(np.float64)
        self._stored_batches = deque(maxlen=self.max_stored_batches) if self.store_recent_data else None
        self._store_batch(X_arr, y_enc, w_arr)
        self.importance_snapshots_ = [
            dict(zip(self._report_names, booster.feature_importance(importance_type="gain").astype(float)))
        ]
        self._importance_engine_cache = None

        metric, loss = self._loss_fn()
        val_loss, n_val = None, 0
        if eval_list:
            X_val = self._prepare_X(eval_list[0][0])
            y_val = self._encode_y(eval_list[0][1])
            val_loss, n_val = loss(y_val, self._raw(booster, X_val)), len(y_val)

        record = self._record(
            stage="fit", action="initial_fit", n_rows=len(y_arr), validation="eval_set" if eval_list else "none",
            n_val_rows=n_val, drift=self.drift_reports_[0], rebin_triggered=False, rebin_accepted=None,
            trees_before=0, trees_retired=0, trees_evicted=0, trees_added=booster.num_trees(),
            trees_after=booster.num_trees(), metric=metric, prequential_loss=None, val_loss_before=None,
            val_loss_after=val_loss, candidate_loss_ours=None, candidate_loss_naive=None,
            window_batches_used=0, train_rows_used=len(y_arr), notes=[], seconds=time.perf_counter() - start,
        )
        self.history_ = [record]
        self._log(
            f"fit: {booster.num_trees()} trees on {len(y_arr)} rows"
            + (f" | val {self._metric_label(metric)} {val_loss:.5g}" if val_loss is not None else "")
        )
        return self

    # ================================================================== partial_fit

    def partial_fit(self, X, y, sample_weight=None, eval_set=None) -> "IncrementalLGBM":
        """Update the model with a new batch (see the module docstring for the algorithm).

        Calling ``partial_fit`` on an unfitted model is the same as ``fit``.
        """
        if not self.__sklearn_is_fitted__():
            return self.fit(X, y, sample_weight=sample_weight, eval_set=eval_set)
        start = time.perf_counter()
        self._validate_params()
        X_arr = self._prepare_X(X)
        y_arr = self._encode_y(y)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]}.")
        w_arr = None if sample_weight is None else as_1d(sample_weight, "sample_weight").astype(np.float64)
        notes: List[str] = []
        metric, loss = self._loss_fn()
        eval_list = _normalize_eval_set(eval_set)
        X_val = y_val = None
        if eval_list:
            X_val = self._prepare_X(eval_list[0][0])
            y_val = self._encode_y(eval_list[0][1])

        # 1. Drift check against the reference distribution.
        drift = self.tracker_.update(X_arr)
        self.drift_reports_.append(drift)
        rebin = self.tracker_.should_rebin(drift)

        current = self.booster_
        n_start = current.num_trees()
        budget = int(math.floor(0.5 * n_start))  # 50% guardrail, shared by all removals in this call
        n_new = self._batch_size()
        # Prequential loss: how well the model predicted this batch *before* seeing it.
        prequential_loss = loss(y_arr, self._raw(current, X_arr))
        val_loss_before = loss(y_val, self._raw(current, X_val)) if eval_list else None

        # 2. Score existing trees on data they have not seen (the eval_set if given,
        #    otherwise the new batch itself), then prune before training.
        score_X, score_y = (X_val, y_val) if eval_list else (X_arr, y_arr)
        tm = TreeManager(current)
        retire: List[int] = []
        scored = False
        try:
            tm.score_trees(score_X, score_y, metric=self.scoring_metric)
            scored = True
        except ValueError as exc:  # e.g. AUC on a single-class validation set
            notes.append(f"tree scoring skipped: {exc}")
        if scored:
            retire = tm.select_retirement(
                self.retirement_threshold, max_count=budget, greedy=True, min_z=_RETIREMENT_MIN_Z
            )
        # Capacity: make room for the new trees within max_total_trees.
        cap_over = max(0, n_start + n_new - self.max_total_trees)
        if cap_over > budget:
            n_new -= cap_over - budget
            cap_over = budget
            notes.append(f"max_total_trees + 50% guardrail: training only {n_new} new trees")
        if scored:
            naive_remove = tm.lowest_scoring(cap_over)
            evict = tm.lowest_scoring(max(0, cap_over - len(retire)), exclude=retire)
        else:  # no scores: evict the oldest trees (tree 0 carries the base score, keep it)
            naive_remove = list(range(1, cap_over + 1))
            evict = naive_remove
        ours_remove = retire + evict
        ours_base = tm.retire_trees(current, ours_remove) if ours_remove else current
        if set(naive_remove) == set(ours_remove):
            naive_base = ours_base
        else:
            naive_base = tm.retire_trees(current, naive_remove) if naive_remove else current

        window = self._compatible_stored_batches(X_arr) if rebin else []

        def train_ours(Xb, yb, wb):
            # Rebin: fresh Dataset (fresh bins) over the current regime's recent data,
            # with the surviving old trees ported in as init_model.
            data = self._rebin_training_data(Xb, yb, wb, stored=window) if rebin else (Xb, yb, wb)
            return self._train(ours_base, *data, n_new), len(data[1])

        def train_naive(Xb, yb, wb):
            return self._train(naive_base, Xb, yb, wb, n_new), len(yb)

        # 3-5. Train. If our update differs from a plain init_model continuation
        # (a rebin, or pruning), check it against that naive candidate.
        differs = rebin or set(ours_remove) != set(naive_remove)
        compare = differs and self.safety_checks
        n_rows = len(y_arr)
        n_hold = int(round(n_rows * self.validation_fraction))
        can_hold_out = (
            self.validation_fraction > 0 and n_hold >= _MIN_VALIDATION_ROWS and n_rows - n_hold >= _MIN_VALIDATION_ROWS
        )
        if eval_list:
            validation = "eval_set"
        elif compare and can_hold_out:
            validation = "holdout"
        else:
            validation = "prequential"
            if compare:
                notes.append("too little data to hold out: update accepted without the naive-baseline safety check")
        cand_ours = cand_naive = None
        use_naive = False

        if validation == "holdout":
            # Select on the most recent rows (walk-forward), then refit the winner on the whole batch.
            cut = n_rows - n_hold
            w_head = None if w_arr is None else w_arr[:cut]
            ours_sel, _ = train_ours(X_arr[:cut], y_arr[:cut], w_head)
            naive_sel, _ = train_naive(X_arr[:cut], y_arr[:cut], w_head)
            use_naive, cand_ours, cand_naive = self._naive_wins(
                y_arr[cut:], self._raw(ours_sel, X_arr[cut:]), self._raw(naive_sel, X_arr[cut:])
            )

        post_retired: List[int] = []
        ours = None
        if use_naive:
            final, rows_used = train_naive(X_arr, y_arr, w_arr)
        else:
            ours, rows_used = train_ours(X_arr, y_arr, w_arr)
            final = ours
            if eval_list:
                # Post-training pruning: the eval_set is honest data for the new trees too.
                remaining_budget = budget - len(ours_remove)
                if remaining_budget > 0 and ours.num_trees() > 0:
                    tm_post = TreeManager(ours)
                    try:
                        tm_post.score_trees(X_val, y_val, metric=self.scoring_metric)
                        post_retired = tm_post.select_retirement(
                            self.retirement_threshold, max_count=remaining_budget, greedy=True, min_z=_RETIREMENT_MIN_Z
                        )
                    except ValueError as exc:
                        notes.append(f"post-training scoring skipped: {exc}")
                    if post_retired:
                        final = tm_post.retire_trees(ours, post_retired)
                if compare:
                    naive, naive_rows = train_naive(X_arr, y_arr, w_arr)
                    use_naive, cand_ours, cand_naive = self._naive_wins(
                        y_val, self._raw(final, X_val), self._raw(naive, X_val)
                    )
                    if use_naive:
                        final, rows_used, post_retired = naive, naive_rows, []

        action = "rebin" if rebin else "incremental"
        if use_naive:
            step = "rebin" if rebin else "pre-training tree retirement"
            label = f"{validation} {self._metric_label(metric)}"
            if cand_ours > cand_naive:
                message = (
                    f"batch {len(self.history_)}: {step} made {label} worse ({cand_ours:.5g} vs {cand_naive:.5g} "
                    "for a plain init_model update); using the plain update instead."
                )
                warnings.warn(message, DriftLGBMWarning, stacklevel=2)
            else:
                message = (
                    f"{step} was not significantly better on {label} ({cand_ours:.5g} vs {cand_naive:.5g}); "
                    "kept the plain init_model update."
                )
            notes.append(message)
            action = "rebin_rejected" if rebin else "retirement_rejected"

        # 6. Book-keeping.
        if not use_naive:
            n_base = ours_base.num_trees()
            retired_set = set(post_retired)
            new_kept = [t for t in range(n_base, ours.num_trees()) if t not in retired_set]
            snapshot_gain = self._gain_of(ours, new_kept)
            trees_retired, trees_evicted, trees_added = len(retire) + len(post_retired), len(evict), len(new_kept)
        else:
            n_base = naive_base.num_trees()
            snapshot_gain = self._gain_of(final, list(range(n_base, final.num_trees())))
            trees_retired, trees_evicted, trees_added = 0, len(naive_remove), final.num_trees() - n_base
        self.booster_ = final
        if rebin:
            # Future drift is measured against the regime the model was just rebuilt for.
            self.tracker_.reset(X_arr)
        self._store_batch(X_arr, y_arr, w_arr)
        self.importance_snapshots_.append(dict(zip(self._report_names, snapshot_gain)))

        record = self._record(
            stage="partial_fit", action=action, n_rows=n_rows, validation=validation,
            n_val_rows=len(y_val) if eval_list else (n_hold if validation == "holdout" else 0), drift=drift,
            rebin_triggered=rebin, rebin_accepted=(action == "rebin") if rebin else None,
            trees_before=n_start, trees_retired=trees_retired, trees_evicted=trees_evicted,
            trees_added=trees_added, trees_after=final.num_trees(), metric=metric,
            prequential_loss=prequential_loss, val_loss_before=val_loss_before,
            val_loss_after=loss(y_val, self._raw(final, X_val)) if eval_list else None,
            candidate_loss_ours=cand_ours, candidate_loss_naive=cand_naive,
            window_batches_used=len(window) if (rebin and not use_naive) else 0,
            train_rows_used=rows_used, notes=notes, seconds=time.perf_counter() - start,
        )
        self.history_.append(record)
        self._log(self._describe(record))
        return self

    # ================================================================== partial_fit helpers

    def _train(self, base: lgb.Booster, X, y, w, n_rounds: int) -> lgb.Booster:
        if n_rounds <= 0:
            return base
        dataset = lgb.Dataset(
            X,
            label=y,
            weight=self._training_weights(y, w),
            feature_name=list(self.feature_name_),
            categorical_feature=self._categorical_feature,
            params=self._train_params,
            free_raw_data=True,
        )
        return lgb.train(
            self._train_params,
            dataset,
            num_boost_round=int(n_rounds),
            init_model=base if base.num_trees() > 0 else None,
            keep_training_booster=False,
        )

    def _compatible_stored_batches(self, X_new: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """Stored batches whose feature distribution still matches the new batch.

        Design decision: a rebin retrains on the *current regime's* recent
        data, not on everything stored. Each stored batch is KS-tested against
        the new batch with the same drift rule as the tracker, and batches
        from a dead regime are left out. Mixing them back in drags the fresh
        trees toward the regime we just detected leaving. Those batches were
        also already fitted by the ported trees, so their residuals are
        partly in-sample noise. On the synthetic regime-change benchmarks,
        retraining on the unfiltered window was measurably worse than a plain
        init_model update. In gradual drift, the recent stored batches still
        match, and they are used.
        """
        stored = list(self._stored_batches) if self._stored_batches is not None else []
        compatible = []
        for batch in stored:
            probe = DistributionTracker(drift_threshold=self.drift_threshold, drift_fraction=self.drift_fraction)
            probe.reset(batch[0])
            if not probe.update(X_new).rebin_recommended:
                compatible.append(batch)
        return compatible

    def _rebin_training_data(self, X_tr, y_tr, w_tr, stored=None):
        stored = list(self._stored_batches or []) if stored is None else list(stored)
        if not stored:
            return X_tr, y_tr, w_tr
        Xs = [b[0] for b in stored] + [X_tr]
        ys = [b[1] for b in stored] + [y_tr]
        if w_tr is None and all(b[2] is None for b in stored):
            w = None
        else:
            w = np.concatenate(
                [b[2] if b[2] is not None else np.ones(len(b[1])) for b in stored]
                + [w_tr if w_tr is not None else np.ones(len(y_tr))]
            )
        return np.vstack(Xs), np.concatenate(ys), w

    def _store_batch(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        if self._stored_batches is None:
            return
        limit = self.max_rows_per_stored_batch
        if limit and len(y) > limit:
            rows = np.sort(self._rng.choice(len(y), size=limit, replace=False))
            X, y = X[rows], y[rows]
            w = None if w is None else w[rows]
        self._stored_batches.append((np.array(X, dtype=np.float64), np.array(y), None if w is None else np.array(w)))

    def _gain_of(self, booster: lgb.Booster, tree_ids: Sequence[int]) -> np.ndarray:
        if not tree_ids:
            return np.zeros(len(self._report_names))
        return per_tree_importance(booster, "gain")[list(tree_ids)].sum(axis=0)

    @staticmethod
    def _metric_label(metric) -> str:
        """Label for *loss* values: losses are lower-is-better, so AUC is reported as 1 - AUC."""
        if not isinstance(metric, str):
            return "loss"
        return "1-auc" if metric == "auc" else metric

    @staticmethod
    def _record(*, drift: DriftReport, **fields) -> Dict[str, Any]:
        record = {
            "batch": drift.batch_index,
            "stage": fields.pop("stage"),
            "action": fields.pop("action"),
            "n_rows": fields.pop("n_rows"),
            "validation": fields.pop("validation"),
            "n_val_rows": fields.pop("n_val_rows"),
            "drift_detected": drift.rebin_recommended,
            "n_drifted_features": drift.n_drifted,
            "fraction_drifted": drift.fraction_drifted,
            "drifted_features": list(drift.drifted_features),
            "p_value_threshold": drift.p_value_threshold,
        }
        record.update(fields)
        record["metric"] = record["metric"] if isinstance(record["metric"], str) else "custom"
        return record

    def _describe(self, r: Dict[str, Any]) -> str:
        label = self._metric_label(r["metric"])
        drift = f"drift {r['fraction_drifted']:.0%} ({r['n_drifted_features']} feat.)"
        parts = [f"batch {r['batch']}", f"{drift} -> {r['action']}"]
        parts.append(
            f"trees {r['trees_before']} -> {r['trees_after']} "
            f"(retired {r['trees_retired']}, evicted {r['trees_evicted']}, added {r['trees_added']})"
        )
        if r["prequential_loss"] is not None:
            parts.append(f"prequential {label} {r['prequential_loss']:.5g}")
        if r["val_loss_after"] is not None:
            parts.append(f"eval {label} {r['val_loss_before']:.5g} -> {r['val_loss_after']:.5g}")
        if r["candidate_loss_naive"] is not None:
            parts.append(
                f"{r['validation']} check: ours {r['candidate_loss_ours']:.5g} vs naive {r['candidate_loss_naive']:.5g}"
            )
        return " | ".join(parts)

    # ================================================================== prediction

    def predict(self, X, raw_score: bool = False) -> np.ndarray:
        """Predict targets (regression), class labels (classification) or raw scores."""
        self._check_fitted()
        X_arr = self._prepare_X(X)
        if raw_score:
            return self._raw(self.booster_, X_arr)
        if self.task == "classification":
            proba = self.booster_.predict(X_arr, num_iteration=-1)
            return self.classes_[(proba > 0.5).astype(np.intp)]
        return self.booster_.predict(X_arr, num_iteration=-1)

    @available_if(lambda self: self.task == "classification")
    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape ``(n_samples, 2)`` (binary classification only)."""
        self._check_fitted()
        proba = self.booster_.predict(self._prepare_X(X), num_iteration=-1)
        return np.column_stack([1.0 - proba, proba])

    def score(self, X, y, sample_weight=None) -> float:
        """R^2 for regression, accuracy for classification (sklearn conventions)."""
        from sklearn.metrics import accuracy_score, r2_score

        if self.task == "classification":
            return float(accuracy_score(y, self.predict(X), sample_weight=sample_weight))
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    # ================================================================== importance

    def feature_importance_report(self, X, y=None) -> ClusteredImportanceReport:
        """Correlation-aware importance of the current model.

        Correlations and clusters are computed on ``X``. Clustered permutation
        importance is included if ``y`` is given, and stability scores come
        from the per-batch importance snapshots gathered during training.
        """
        self._check_fitted()
        frame = pd.DataFrame(self._prepare_X(X), columns=self._report_names)
        y_enc = None if y is None else self._encode_y(y)
        return self._importance_engine().compute(self.booster_, frame, y=y_enc, snapshots=self.importance_snapshots_)

    def _importance_engine(self) -> ClusteredImportance:
        key = (self.correlation_method, self.cluster_distance_threshold)
        cached = getattr(self, "_importance_engine_cache", None)
        if cached is None or cached[0] != key:
            seed = self._train_params.get("random_state") if hasattr(self, "_train_params") else None
            engine = ClusteredImportance(
                method=self.correlation_method,
                distance_threshold=self.cluster_distance_threshold,
                random_state=seed if isinstance(seed, (int, np.integer)) else 0,
            )
            self._importance_engine_cache = (key, engine)
        return self._importance_engine_cache[1]
