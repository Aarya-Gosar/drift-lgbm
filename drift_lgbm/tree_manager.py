"""Per-tree contribution scoring and tree retirement for LightGBM boosters.

A boosted ensemble is additive in raw-score space: ``F(x) = sum_j tree_j(x)``.
LightGBM folds the initial score (``boost_from_average``) into the first tree
and the learning rate into every leaf value. So removing tree ``j`` gives
exactly ``F(x) - tree_j(x)``, and leave-one-out (LOO) scoring of every tree
needs just one pass over the per-tree outputs, not one refit per tree.

Per-tree outputs are computed in one shot: ``predict(pred_leaf=True)`` gives
each row's leaf in each tree, and those indices are looked up in the leaf
values parsed from the model text. Linear trees, whose leaves are not
constants, fall back to one ``predict`` call per iteration.

.. warning::
   :meth:`TreeManager.retire_trees` edits LightGBM's model *text* format,
   which is undocumented. This is a deliberate hack: it cuts the chosen
   ``Tree=`` blocks out of ``model_to_string()``, renumbers the rest,
   recomputes the ``tree_sizes`` header (LightGBM hard-crashes on a wrong
   one), and reloads the text with ``lgb.Booster(model_str=...)``. It is
   verified against LightGBM 4.x (``version=v4``) and pinned by the test
   suite, and the result is checked for the expected tree count. If a future
   LightGBM changes the format, the fallback is to keep a mask of active trees
   and subtract retired trees' outputs at predict time.

Only single-output models are supported: regression-style objectives and
binary classification (``num_tree_per_iteration == 1``), boosting modes that
sum their trees (gbdt, goss, dart). Random-forest mode averages its trees, so
LOO-by-subtraction does not apply there, and it is rejected. Instances are
not thread-safe.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
from scipy import stats

from ._utils import align_columns, as_1d, as_2d_float

__all__ = [
    "TreeManager",
    "per_tree_importance",
    "booster_objective",
    "loss_function",
    "row_loss_function",
    "resolve_metric",
    "compute_metric",
]

# Cap on the number of float64 cells materialised at once when computing
# per-tree outputs (rows x trees); ~32 MB.
_MAX_BLOCK_CELLS = 4_000_000

_METRIC_ALIASES = {
    "rmse": "rmse",
    "l2_root": "rmse",
    "root_mean_squared_error": "rmse",
    "mae": "mae",
    "l1": "mae",
    "mean_absolute_error": "mae",
    "logloss": "logloss",
    "log_loss": "logloss",
    "binary_logloss": "logloss",
    "binary": "logloss",
    "cross_entropy": "logloss",
    "auc": "auc",
    "roc_auc": "auc",
}
_HIGHER_IS_BETTER = {"auc"}


# --------------------------------------------------------------------------- model text


class _ModelText:
    """A LightGBM model string split into header, per-tree blocks and footer."""

    _TREE_START = re.compile(r"^Tree=\d+$", re.M)

    def __init__(self, text: str):
        end = text.find("end of trees")
        if not text.startswith("tree") or end < 0:
            raise ValueError("Unrecognised LightGBM model text (missing 'tree' header or 'end of trees').")
        starts = [m.start() for m in self._TREE_START.finditer(text, 0, end)]
        self.header = text[: starts[0]] if starts else text[:end]
        self.trees = [text[s:e] for s, e in zip(starts, starts[1:] + [end])]
        self.footer = text[end:]

        version = self.header_value("version")
        if version not in ("v3", "v4"):
            warnings.warn(
                f"LightGBM model text version {version!r} is untested with drift-lgbm's tree surgery.",
                RuntimeWarning,
                stacklevel=3,
            )

    def header_value(self, key: str) -> Optional[str]:
        match = re.search(rf"^{re.escape(key)}=(.*)$", self.header, re.M)
        return match.group(1) if match else None

    def has_header_flag(self, flag: str) -> bool:
        return re.search(rf"^{re.escape(flag)}$", self.header, re.M) is not None

    def objective(self) -> Tuple[str, Dict[str, str]]:
        raw = (self.header_value("objective") or "regression").split()
        options = dict(tok.split(":", 1) for tok in raw[1:] if ":" in tok)
        return raw[0], options

    def feature_names(self) -> List[str]:
        return (self.header_value("feature_names") or "").split()

    @staticmethod
    def _field(block: str, key: str) -> str:
        match = re.search(rf"^{key}=(.*)$", block, re.M)
        return match.group(1).strip() if match else ""

    def leaf_values(self) -> List[np.ndarray]:
        return [np.array(self._field(t, "leaf_value").split(), dtype=np.float64) for t in self.trees]

    def splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Per tree: (split feature indices, split gains)."""
        out = []
        for t in self.trees:
            feats = np.array(self._field(t, "split_feature").split(), dtype=np.int64)
            gains = np.array(self._field(t, "split_gain").split(), dtype=np.float64)
            out.append((feats, gains))
        return out

    def has_linear_trees(self) -> bool:
        return any(self._field(t, "is_linear") == "1" for t in self.trees)

    def render(self, keep: Sequence[int]) -> str:
        """Model text containing only the trees in ``keep`` (in that order), renumbered."""
        blocks = [
            self._TREE_START.sub(f"Tree={new_idx}", self.trees[old_idx], count=1)
            for new_idx, old_idx in enumerate(keep)
        ]
        sizes = " ".join(str(len(b.encode("utf-8"))) for b in blocks)
        if re.search(r"^tree_sizes=", self.header, re.M):
            header = re.sub(r"^tree_sizes=.*$", f"tree_sizes={sizes}", self.header, count=1, flags=re.M)
        else:
            header = self.header
        if not blocks:
            header = re.sub(r"^tree_sizes=.*\n", "", header, count=1, flags=re.M)
        return header + "".join(blocks) + self.footer


def _check_supported(model: _ModelText) -> None:
    per_iter = int(model.header_value("num_tree_per_iteration") or 1)
    if per_iter != 1:
        raise NotImplementedError(
            "TreeManager supports single-output models only (regression / binary); "
            f"got num_tree_per_iteration={per_iter}."
        )
    if model.has_header_flag("average_output"):
        raise NotImplementedError("Random-forest boosters average their trees; LOO scoring by subtraction does not apply.")


def per_tree_importance(booster: lgb.Booster, importance_type: str = "gain") -> np.ndarray:
    """Feature importance of every individual tree, shape ``(n_trees, n_features)``.

    Summing over rows reproduces ``booster.feature_importance(importance_type)``.
    Summing over a subset of rows gives the importance of just those trees,
    e.g. the trees added by one incremental batch.
    """
    if importance_type not in ("gain", "split"):
        raise ValueError("importance_type must be 'gain' or 'split'.")
    model = _ModelText(booster.model_to_string())
    n_features = booster.num_feature()
    out = np.zeros((len(model.trees), n_features))
    for t, (feats, gains) in enumerate(model.splits()):
        weights = gains if importance_type == "gain" else np.ones_like(gains)
        np.add.at(out[t], feats, weights)
    return out


# --------------------------------------------------------------------------- metrics


def resolve_metric(metric: Optional[Union[str, Callable]], objective: str) -> Union[str, Callable]:
    """Canonical metric name, or the callable itself. ``None`` picks a default for the objective."""
    if callable(metric):
        return metric
    if metric is None:
        return "logloss" if objective in ("binary", "cross_entropy", "xentropy") else "rmse"
    key = str(metric).lower()
    if key not in _METRIC_ALIASES:
        raise ValueError(f"Unknown metric {metric!r}; choose from rmse, mae, logloss, auc or pass a callable.")
    return _METRIC_ALIASES[key]


def _prediction_transform(objective: str, options: Dict[str, str]) -> Callable[[np.ndarray], np.ndarray]:
    """Map raw scores to the prediction scale for a given objective."""
    if objective == "binary":
        sigmoid = float(options.get("sigmoid", 1.0))
        return lambda raw: 1.0 / (1.0 + np.exp(-sigmoid * raw))
    if objective in ("cross_entropy", "xentropy"):
        return lambda raw: 1.0 / (1.0 + np.exp(-raw))
    if objective in ("poisson", "gamma", "tweedie"):
        return np.exp
    return lambda raw: raw


def compute_metric(
    metric: Union[str, Callable],
    y: np.ndarray,
    raw: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    sigmoid: float = 1.0,
) -> np.ndarray:
    """Evaluate ``metric`` for one or many raw-score columns.

    ``raw`` has shape ``(n,)`` or ``(n, k)``. Returns a scalar array or an array
    of ``k`` values. Log-loss is computed from raw scores directly
    (``softplus(z) - y*z``), which is numerically stable.
    """
    y_col = y if raw.ndim == 1 else y[:, None]
    if callable(metric):
        if raw.ndim == 1:
            return np.asarray(metric(y, transform(raw)), dtype=np.float64)
        return np.array([metric(y, transform(raw[:, k])) for k in range(raw.shape[1])], dtype=np.float64)
    if metric == "rmse":
        return np.sqrt(np.mean((y_col - transform(raw)) ** 2, axis=0))
    if metric == "mae":
        return np.mean(np.abs(y_col - transform(raw)), axis=0)
    if metric == "logloss":
        z = sigmoid * raw
        return np.mean(np.logaddexp(0.0, z) - y_col * z, axis=0)
    if metric == "auc":
        n_pos = float(np.sum(y == 1))
        n_neg = float(y.shape[0] - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("AUC is undefined when y_val contains a single class.")
        ranks = stats.rankdata(raw, axis=0)
        pos_rank_sum = ranks[y == 1].sum(axis=0)
        return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    raise ValueError(f"Unknown metric {metric!r}.")


def _per_row_loss(metric, y: np.ndarray, raw: np.ndarray, transform, sigmoid: float) -> Optional[np.ndarray]:
    """Per-row loss for metrics that are means over rows (rmse via squared error); else None."""
    if metric == "rmse":
        return (y - transform(raw)) ** 2
    if metric == "mae":
        return np.abs(y - transform(raw))
    if metric == "logloss":
        z = sigmoid * raw
        return np.logaddexp(0.0, z) - y * z
    return None


def booster_objective(booster: lgb.Booster) -> Tuple[str, Dict[str, str]]:
    """``(objective_name, options)`` parsed from the booster's model header."""
    # Only the header is needed; serialising a single tree keeps this cheap.
    return _ModelText(booster.model_to_string(num_iteration=1)).objective()


def loss_function(
    objective: str,
    options: Optional[Dict[str, str]] = None,
    metric: Optional[Union[str, Callable]] = None,
) -> Tuple[Union[str, Callable], Callable[[np.ndarray, np.ndarray], float]]:
    """Return ``(metric, loss)`` where ``loss(y, raw_scores)`` is lower-is-better.

    AUC is turned into a loss as ``1 - AUC``, so every guardrail comparison
    can use "smaller is better".
    """
    options = options or {}
    metric = resolve_metric(metric, objective)
    transform = _prediction_transform(objective, options)
    sigmoid = float(options.get("sigmoid", 1.0)) if objective == "binary" else 1.0

    def loss(y: np.ndarray, raw: np.ndarray) -> float:
        value = float(compute_metric(metric, np.asarray(y, dtype=np.float64), raw, transform, sigmoid))
        return 1.0 - value if metric in _HIGHER_IS_BETTER else value

    return metric, loss


def row_loss_function(
    objective: str,
    options: Optional[Dict[str, str]] = None,
    metric: Optional[Union[str, Callable]] = None,
) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Per-row loss ``f(y, raw_scores) -> array``, for metrics that average over
    rows (rmse via squared error, mae, logloss). ``None`` for AUC and callables."""
    options = options or {}
    metric = resolve_metric(metric, objective)
    if metric not in ("rmse", "mae", "logloss"):
        return None
    transform = _prediction_transform(objective, options)
    sigmoid = float(options.get("sigmoid", 1.0)) if objective == "binary" else 1.0
    return lambda y, raw: _per_row_loss(metric, np.asarray(y, dtype=np.float64), raw, transform, sigmoid)


# --------------------------------------------------------------------------- manager


class TreeManager:
    """Score the trees of a LightGBM booster and retire the ones that hurt.

    Parameters
    ----------
    booster : lightgbm.Booster, optional
        The booster to manage. It can also be passed per call.
    max_retire_fraction : float, default=0.5
        Safety guardrail: one :meth:`retire_trees` call never removes more than
        this fraction of the ensemble.

    Attributes (after :meth:`score_trees`)
    ----------------------------------------
    scores_ : ndarray of shape (n_trees,)
        Contribution of each tree, indexed by tree position. Positive means the
        tree helps: removing it makes the metric worse.
    metric_ : str or callable
        The metric the scores were computed with.
    baseline_ : float
        Metric value of the full ensemble on the scoring set.
    """

    def __init__(self, booster: Optional[lgb.Booster] = None, max_retire_fraction: float = 0.5):
        if not 0.0 <= max_retire_fraction <= 1.0:
            raise ValueError("max_retire_fraction must be in [0, 1].")
        self.booster = booster
        self.max_retire_fraction = float(max_retire_fraction)
        self.scores_: Optional[np.ndarray] = None
        self.metric_ = None
        self.baseline_: Optional[float] = None
        self._eval = None  # cached scoring context, reused by select_retirement()

    # -------------------------------------------------------------- scoring

    def score_trees(
        self,
        X_val,
        y_val,
        metric: Optional[Union[str, Callable]] = None,
        booster: Optional[lgb.Booster] = None,
    ) -> List[Tuple[int, float]]:
        """Leave-one-out contribution of every tree on ``(X_val, y_val)``.

        For each tree ``j`` the prediction without it is ``F - tree_j``, and
        its contribution is how much the metric worsens when it is removed:
        ``loss(F - tree_j) - loss(F)`` for losses (rmse, mae, logloss,
        callables), and ``auc(F) - auc(F - tree_j)`` for AUC. Negative
        contributions mark trees that hurt.

        A callable ``metric(y_true, y_pred)`` is treated as a loss (lower is
        better) and gets predictions on the objective's output scale. It is
        evaluated once per tree, so it is much slower than the built-ins.

        Returns ``(tree_index, contribution)`` pairs sorted from most helpful
        to most harmful. The harmful trees are at the bottom.
        """
        booster = self._resolve_booster(booster)
        model = _ModelText(booster.model_to_string())
        _check_supported(model)
        objective, options = model.objective()
        metric = resolve_metric(metric, objective)
        transform = _prediction_transform(objective, options)
        sigmoid = float(options.get("sigmoid", 1.0)) if objective == "binary" else 1.0

        X = self._prepare_X(X_val, booster)
        y = as_1d(y_val, "y_val").astype(np.float64)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X_val and y_val have different numbers of rows.")
        if metric in ("logloss", "auc") and not np.all((y >= 0) & (y <= 1)):
            raise ValueError(f"metric={metric!r} expects labels in [0, 1] (encode classes as 0/1).")

        n_trees = booster.num_trees()
        full = booster.predict(X, raw_score=True, num_iteration=-1) if n_trees else np.zeros(X.shape[0])
        baseline = float(compute_metric(metric, y, full, transform, sigmoid))
        sign = -1.0 if metric in _HIGHER_IS_BETTER else 1.0

        scores = np.zeros(n_trees)
        use_leaf_lookup = not model.has_linear_trees()
        leaf_values = model.leaf_values() if use_leaf_lookup else None
        check_sum = np.zeros(X.shape[0])
        for start, block in self._iter_tree_outputs(booster, X, range(n_trees), leaf_values):
            check_sum += block.sum(axis=1)
            without = full[:, None] - block
            values = compute_metric(metric, y, without, transform, sigmoid)
            scores[start : start + block.shape[1]] = sign * (values - baseline)

        if n_trees and not np.allclose(check_sum, full, rtol=1e-6, atol=1e-6):
            # Parsed leaf values do not reproduce LightGBM's own prediction: the
            # model text is not what we expect. Recompute with the slow path.
            if use_leaf_lookup:
                warnings.warn(
                    "Per-tree outputs from the parsed model text do not sum to the booster's raw "
                    "prediction; falling back to per-iteration prediction.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                leaf_values = None
                for start, block in self._iter_tree_outputs(booster, X, range(n_trees), None):
                    values = compute_metric(metric, y, full[:, None] - block, transform, sigmoid)
                    scores[start : start + block.shape[1]] = sign * (values - baseline)
            else:  # pragma: no cover - would indicate a LightGBM bug
                raise RuntimeError("Per-iteration predictions do not sum to the full raw prediction.")

        self.booster = booster
        self.scores_ = scores
        self.metric_ = metric
        self.baseline_ = baseline
        self._eval = {
            "X": X,
            "y": y,
            "full": full,
            "transform": transform,
            "sigmoid": sigmoid,
            "sign": sign,
            "leaf_values": leaf_values,
            "booster": booster,
        }
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order]

    def identify_retiring(self, threshold: float = 0.0) -> List[int]:
        """Trees whose contribution is below ``threshold``, most harmful first.

        The default threshold 0 selects trees whose removal improves the
        metric. This does not apply the 50% guardrail; :meth:`retire_trees`
        does.
        """
        scores = self._require_scores()
        candidates = np.flatnonzero(scores < threshold)
        return [int(i) for i in candidates[np.argsort(scores[candidates], kind="stable")]]

    def select_retirement(
        self,
        threshold: float = 0.0,
        max_fraction: Optional[float] = None,
        max_count: Optional[int] = None,
        greedy: bool = True,
        min_z: float = 0.0,
    ) -> List[int]:
        """Choose trees to retire, capped and (optionally) verified jointly.

        LOO scores are marginal: two correlated harmful trees can each look
        bad alone and still be worth keeping together. With ``greedy=True``
        the candidates from :meth:`identify_retiring` are taken most harmful
        first, and each is kept only if removing it *on top of the removals
        already accepted* strictly improves the scoring-set metric. The final
        selection is therefore never worse than the full ensemble on that set.

        ``min_z > 0`` also requires each removal to be *significant*. For
        metrics that average over rows (rmse, mae, logloss), the mean per-row
        loss change must be below ``-min_z`` standard errors. Without it, a
        tree whose true effect is ~0 is retired whenever validation noise
        happens to favour its removal, which overfits the validation set.
        AUC and callables fall back to strict improvement.
        """
        scores = self._require_scores()
        n_trees = scores.shape[0]
        fraction = self.max_retire_fraction if max_fraction is None else max_fraction
        cap = int(math.floor(fraction * n_trees))
        if max_count is not None:
            cap = min(cap, int(max_count))
        candidates = self.identify_retiring(threshold)
        if not greedy or not candidates or cap <= 0:
            return candidates[: max(cap, 0)]

        ev = self._eval
        y, transform, sigmoid = ev["y"], ev["transform"], ev["sigmoid"]
        current = ev["full"].copy()
        current_value = float(compute_metric(self.metric_, y, current, transform, sigmoid))
        current_rows = _per_row_loss(self.metric_, y, current, transform, sigmoid) if min_z > 0 else None
        n = y.shape[0]
        selected: List[int] = []
        for j in candidates:
            if len(selected) >= cap:
                break
            _, block = next(self._iter_tree_outputs(ev["booster"], ev["X"], [j], ev["leaf_values"]))
            trial = current - block[:, 0]
            if current_rows is not None:
                trial_rows = _per_row_loss(self.metric_, y, trial, transform, sigmoid)
                diff = trial_rows - current_rows
                std_err = float(np.std(diff, ddof=1)) / math.sqrt(n) if n > 1 else 0.0
                accept = diff.mean() < 0 and diff.mean() < -min_z * std_err
            else:
                value = float(compute_metric(self.metric_, y, trial, transform, sigmoid))
                accept = ev["sign"] * (value - current_value) < 0  # strictly better
            if accept:
                current = trial
                current_value = float(compute_metric(self.metric_, y, current, transform, sigmoid))
                if current_rows is not None:
                    current_rows = trial_rows
                selected.append(j)
        return selected

    def lowest_scoring(self, n: int, exclude: Sequence[int] = ()) -> List[int]:
        """The ``n`` lowest-contribution trees (ignoring ``exclude``), lowest first."""
        scores = self._require_scores()
        excluded = set(int(i) for i in exclude)
        order = [int(i) for i in np.argsort(scores, kind="stable") if int(i) not in excluded]
        return order[: max(int(n), 0)]

    # -------------------------------------------------------------- retirement

    def retire_trees(self, booster: Optional[lgb.Booster], indices: Sequence[int]) -> lgb.Booster:
        """Return a NEW booster with the trees at ``indices`` removed.

        The input booster is never modified. If ``indices`` asks for more than
        ``max_retire_fraction`` of the trees, only the first ones in the given
        order are retired (so pass them most harmful first, as
        :meth:`identify_retiring` does) and a warning is issued.

        Implementation note: this edits LightGBM's undocumented model text
        (see the module docstring). It is hacky, but it gives a real, smaller
        ``Booster`` that predicts and continues training like any other.
        """
        booster = self._resolve_booster(booster)
        n_trees = booster.num_trees()
        requested = [int(i) for i in indices]
        if len(set(requested)) != len(requested):
            raise ValueError("Duplicate tree indices passed to retire_trees.")
        bad = [i for i in requested if not 0 <= i < n_trees]
        if bad:
            raise ValueError(f"Tree indices out of range for a {n_trees}-tree booster: {bad[:10]}")

        max_retire = int(math.floor(self.max_retire_fraction * n_trees))
        if len(requested) > max_retire:
            warnings.warn(
                f"Requested retiring {len(requested)} of {n_trees} trees; the guardrail caps a single "
                f"call at {self.max_retire_fraction:.0%} ({max_retire} trees). Retiring the first "
                f"{max_retire} in the given order.",
                UserWarning,
                stacklevel=2,
            )
            requested = requested[:max_retire]

        model = _ModelText(booster.model_to_string())
        _check_supported(model)
        if len(model.trees) != n_trees:
            raise RuntimeError(f"Parsed {len(model.trees)} tree blocks but the booster reports {n_trees} trees.")
        removed = set(requested)
        keep = [i for i in range(n_trees) if i not in removed]
        new_booster = lgb.Booster(model_str=model.render(keep))
        if new_booster.num_trees() != len(keep):
            raise RuntimeError(
                f"Tree surgery produced {new_booster.num_trees()} trees, expected {len(keep)}; "
                "the LightGBM model text format may have changed."
            )
        return new_booster

    # -------------------------------------------------------------- helpers

    def _resolve_booster(self, booster: Optional[lgb.Booster]) -> lgb.Booster:
        booster = booster if booster is not None else self.booster
        if booster is None:
            raise ValueError("No booster given (pass one to TreeManager(...) or to the method).")
        if hasattr(booster, "booster_") and not isinstance(booster, lgb.Booster):
            booster = booster.booster_
        if not isinstance(booster, lgb.Booster):
            raise TypeError(f"Expected a lightgbm.Booster, got {type(booster).__name__}.")
        return booster

    def _require_scores(self) -> np.ndarray:
        if self.scores_ is None:
            raise RuntimeError("Call score_trees() first.")
        return self.scores_

    @staticmethod
    def _prepare_X(X_val, booster: lgb.Booster) -> np.ndarray:
        arr, names = as_2d_float(X_val)
        expected = booster.feature_name()
        expected_set = set(expected)
        has_names = names is not None and all(str(n) in expected_set for n in names)
        return align_columns(arr, [str(n) for n in names] if has_names else None, expected, has_names)

    @staticmethod
    def _iter_tree_outputs(booster, X, tree_indices, leaf_values):
        """Yield ``(first_tree_index, outputs)`` blocks of per-tree raw outputs.

        ``tree_indices`` must be contiguous ascending (a ``range``) or a single
        index. Each block covers consecutive trees and has shape ``(n, k)``.
        """
        tree_indices = list(tree_indices)
        if not tree_indices:
            return
        n = X.shape[0]
        block_size = max(1, _MAX_BLOCK_CELLS // max(n, 1))
        first, last = tree_indices[0], tree_indices[-1] + 1
        for start in range(first, last, block_size):
            stop = min(start + block_size, last)
            if leaf_values is not None:
                leaves = booster.predict(
                    X, pred_leaf=True, start_iteration=start, num_iteration=stop - start
                ).reshape(n, -1).astype(np.intp, copy=False)
                block = np.empty((n, stop - start))
                for k, t in enumerate(range(start, stop)):
                    block[:, k] = leaf_values[t][leaves[:, k]]
            else:
                block = np.column_stack(
                    [booster.predict(X, raw_score=True, start_iteration=t, num_iteration=1) for t in range(start, stop)]
                )
            yield start, block

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_eval"] = None  # scoring cache holds the validation data; not worth pickling
        return state
