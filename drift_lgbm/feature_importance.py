"""Correlation-aware (clustered) feature importance.

Standard gain importance splits credit arbitrarily between correlated
features. If the model can use ``momentum_10d`` or ``momentum_15d``
interchangeably, the split between them is noise, and dropping one roughly
doubles the other's importance. Following Lopez de Prado (*Advances in
Financial Machine Learning*, ch. 8), we cluster the features by dependence and
report importance per *cluster*, which is stable under such substitutions.
Feature selection should happen at the cluster level.

Pipeline:

1. **Dependence matrix**: Spearman rank correlation by default. Pearson and
   pairwise mutual information are also available.
2. **Clustering**: ``distance = 1 - |dependence|``, then agglomerative
   clustering (``scipy.cluster.hierarchy``) cut at ``distance_threshold``.
   With the default threshold 0.5, features whose |correlation| exceeds
   ~0.5 end up together. How strictly depends on ``linkage``: average by
   default.
3. **Cluster importance**: the sum of the members' LightGBM importances (gain
   by default). The sum over clusters therefore equals the total importance.
4. **Intra-cluster share**: each member's fraction of its cluster's
   importance. It is informational only, since the split *within* a cluster
   is the arbitrary part.
5. **Stability**: given importance snapshots from successive training
   batches, a per-cluster decomposition of Kendall's tau measures how
   consistently each cluster keeps its rank relative to the others.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from ._utils import as_1d, as_2d_float
from .tree_manager import booster_objective, loss_function

__all__ = ["ClusteredImportance", "ClusteredImportanceReport"]

_VALID_METHODS = ("spearman", "pearson", "mutual_info")
_VALID_LINKAGES = ("average", "complete", "single", "weighted")


@dataclass
class ClusteredImportanceReport:
    """Cluster-level feature importance.

    Attributes
    ----------
    clusters : dict
        cluster_id -> list of feature names. Ids are ``0..K-1``, numbered by
        each cluster's first feature in column order, so they are stable
        across reports that use the same clustering.
    cluster_importance : dict
        cluster_id -> summed raw importance of its members.
    feature_importance : dict
        feature_name -> ``(cluster_id, intra_cluster_share)``. The shares
        within a cluster sum to 1.
    stability_scores : dict or None
        cluster_id -> stability in ``[-1, 1]`` (see
        :meth:`ClusteredImportance.stability`). ``None`` unless at least two
        importance snapshots were given.
    correlation_matrix : pandas.DataFrame
        The dependence matrix the clustering used. For ``method='mutual_info'``
        it holds Linfoot's informational coefficient of correlation, which
        lies in ``[0, 1]``.
    raw_feature_importance : dict
        feature_name -> the model's raw importance.
    cluster_permutation_importance : dict or None
        cluster_id -> increase in validation loss when the whole cluster is
        permuted jointly (clustered MDA). Only present when ``y`` was given.
    overall_stability : float or None
        Mean Kendall's tau-b between consecutive snapshots' cluster importances.
    """

    clusters: Dict[int, List[Hashable]]
    cluster_importance: Dict[int, float]
    feature_importance: Dict[Hashable, Tuple[int, float]]
    stability_scores: Optional[Dict[int, float]]
    correlation_matrix: pd.DataFrame
    method: str = "spearman"
    distance_threshold: float = 0.5
    linkage: str = "average"
    importance_type: str = "gain"
    raw_feature_importance: Optional[Dict[Hashable, float]] = None
    cluster_permutation_importance: Optional[Dict[int, float]] = None
    permutation_metric: Optional[str] = None
    overall_stability: Optional[float] = None
    n_snapshots: int = 0

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def total_importance(self) -> float:
        return float(sum(self.cluster_importance.values()))

    @property
    def cluster_share(self) -> Dict[int, float]:
        """cluster_id -> fraction of the total importance."""
        total = self.total_importance
        return {c: (v / total if total > 0 else 0.0) for c, v in self.cluster_importance.items()}

    def cluster_of(self, feature: Hashable) -> int:
        return self.feature_importance[feature][0]

    def to_frame(self) -> pd.DataFrame:
        """One row per cluster, sorted by importance."""
        share = self.cluster_share
        rows = []
        for cid, members in self.clusters.items():
            ordered = sorted(members, key=lambda f: -self.feature_importance[f][1])
            rows.append(
                {
                    "cluster": cid,
                    "importance": self.cluster_importance[cid],
                    "share": share[cid],
                    "n_features": len(members),
                    "stability": (self.stability_scores or {}).get(cid, np.nan),
                    "permutation_importance": (self.cluster_permutation_importance or {}).get(cid, np.nan),
                    "features": ordered,
                }
            )
        frame = pd.DataFrame(rows).set_index("cluster")
        return frame.sort_values("importance", ascending=False)

    def feature_frame(self) -> pd.DataFrame:
        """One row per feature: cluster, intra-cluster share and raw importance."""
        raw = self.raw_feature_importance or {}
        frame = pd.DataFrame(
            {
                "cluster": {f: c for f, (c, _) in self.feature_importance.items()},
                "intra_cluster_share": {f: s for f, (_, s) in self.feature_importance.items()},
                "raw_importance": {f: raw.get(f, np.nan) for f in self.feature_importance},
            }
        )
        frame.index.name = "feature"
        return frame.sort_values(["cluster", "intra_cluster_share"], ascending=[True, False])

    def to_string(self, max_clusters: int = 20, max_features: int = 4) -> str:
        frame = self.to_frame()
        header = (
            f"Clustered feature importance (method={self.method}, distance_threshold="
            f"{self.distance_threshold}, linkage={self.linkage}, importance={self.importance_type})"
        )
        sub = f"{len(self.feature_importance)} features in {self.n_clusters} clusters; total {self.importance_type} {self.total_importance:,.1f}"
        if self.stability_scores is not None:
            overall = "n/a" if self.overall_stability is None else f"{self.overall_stability:+.2f}"
            sub += f"; {self.n_snapshots} snapshots, overall stability (Kendall tau) {overall}"
        lines = [header, sub, ""]
        has_stab = self.stability_scores is not None
        has_perm = self.cluster_permutation_importance is not None
        cols = f"{'#':>3} {'cluster':>7} {'share':>7} {'importance':>12}"
        if has_stab:
            cols += f" {'stability':>9}"
        if has_perm:
            cols += f" {'perm.' + (self.permutation_metric or ''):>12}"
        lines.append(cols + "  members (intra-cluster share)")
        for rank, (cid, row) in enumerate(frame.head(max_clusters).iterrows(), start=1):
            members = ", ".join(
                f"{f} ({self.feature_importance[f][1]:.0%})" for f in row["features"][:max_features]
            )
            if len(row["features"]) > max_features:
                members += f", +{len(row['features']) - max_features} more"
            line = f"{rank:>3} {cid:>7} {row['share']:>7.1%} {row['importance']:>12,.1f}"
            if has_stab:
                line += f" {row['stability']:>+9.2f}"
            if has_perm:
                line += f" {row['permutation_importance']:>12.4g}"
            lines.append(line + "  " + members)
        if len(frame) > max_clusters:
            rest = frame.iloc[max_clusters:]
            lines.append(f"... {len(rest)} more clusters holding {rest['share'].sum():.1%} of importance")
        return "\n".join(lines)

    def summary(self, max_clusters: int = 20, max_features: int = 4) -> None:
        """Print a human-readable report."""
        print(self.to_string(max_clusters=max_clusters, max_features=max_features))

    def __str__(self) -> str:
        return self.to_string()


def _extract_booster(model) -> lgb.Booster:
    """Accept a Booster, a fitted LGBMModel, or anything else exposing ``booster_``."""
    if isinstance(model, lgb.Booster):
        return model
    booster = getattr(model, "booster_", None)
    if isinstance(booster, lgb.Booster):
        return booster
    raise TypeError(
        f"Expected a lightgbm.Booster or a fitted model with a `booster_` attribute, got {type(model).__name__}."
    )


class ClusteredImportance:
    """Compute correlation-aware, cluster-level feature importance.

    Parameters
    ----------
    method : {'spearman', 'pearson', 'mutual_info'}, default='spearman'
        Dependence measure. Spearman is the default because financial features
        are often monotonically but non-linearly related. ``'mutual_info'``
        also catches non-monotonic dependence (e.g. ``x`` and ``x**2``). It
        estimates pairwise MI with ``sklearn.feature_selection.mutual_info_regression``
        and maps it to ``[0, 1]`` with Linfoot's informational coefficient of
        correlation, ``sqrt(1 - exp(-2 * MI))``, which equals ``|rho|`` for
        Gaussian data, so the same ``distance_threshold`` means the same
        thing. It is O(n_features^2) and slow: with more than
        ``mi_max_features`` features we warn and fall back to Spearman.
    distance_threshold : float, default=0.5
        The main hyper-parameter. Clusters are cut where the linkage distance
        (``1 - |dependence|``) exceeds it. Near 0 every feature is its own
        cluster, which reduces to plain importance. Near 1 everything merges
        into one cluster, which tells you nothing.
    linkage : {'average', 'complete', 'single', 'weighted'}, default='average'
        Agglomerative linkage. ``'single'`` merges any pair above the
        correlation cut, transitively, and can chain loosely related features
        together. ``'complete'`` requires every pair to be close. ``'average'``
        (the default) sits between the two.
    importance_type : {'gain', 'split'}, default='gain'
        LightGBM importance type to aggregate.
    max_samples : int, default=50_000
        Rows used to estimate correlations (random subsample beyond this).
    mi_max_samples : int, default=2_000
        Rows used for mutual-information estimation.
    mi_max_features : int, default=500
        Above this many features ``'mutual_info'`` falls back to Spearman.
    n_repeats : int, default=3
        Repeats for clustered permutation importance, when ``y`` is given.
    permutation_max_samples : int, default=10_000
        Rows used for permutation importance.
    random_state : int, default=0
        Seed for subsampling and permutations.
    cache_size : int, default=4
        How many dependence matrices to keep in the (content-hashed) cache.
        Computing them is the expensive part for wide datasets.
    """

    def __init__(
        self,
        method: str = "spearman",
        distance_threshold: float = 0.5,
        linkage: str = "average",
        importance_type: str = "gain",
        max_samples: int = 50_000,
        mi_max_samples: int = 2_000,
        mi_max_features: int = 500,
        n_repeats: int = 3,
        permutation_max_samples: int = 10_000,
        random_state: int = 0,
        cache_size: int = 4,
    ):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}.")
        if linkage not in _VALID_LINKAGES:
            raise ValueError(f"linkage must be one of {_VALID_LINKAGES}, got {linkage!r}.")
        if importance_type not in ("gain", "split"):
            raise ValueError("importance_type must be 'gain' or 'split'.")
        if not 0.0 <= distance_threshold <= 1.0:
            raise ValueError("distance_threshold must be in [0, 1] (distance = 1 - |correlation|).")
        self.method = method
        self.distance_threshold = float(distance_threshold)
        self.linkage = linkage
        self.importance_type = importance_type
        self.max_samples = int(max_samples)
        self.mi_max_samples = int(mi_max_samples)
        self.mi_max_features = int(mi_max_features)
        self.n_repeats = int(n_repeats)
        self.permutation_max_samples = int(permutation_max_samples)
        self.random_state = random_state
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
        self.cache_hits_ = 0
        self.cache_misses_ = 0

    # ------------------------------------------------------------------ dependence

    def correlation(self, X, feature_names: Optional[Sequence[Hashable]] = None) -> pd.DataFrame:
        """Dependence matrix of ``X`` (cached by content), as a labelled DataFrame."""
        arr, names = as_2d_float(X)
        names = list(names if names is not None else (feature_names or range(arr.shape[1])))
        if len(names) != arr.shape[1]:
            raise ValueError("feature_names length does not match the number of columns.")
        method = self._effective_method(arr.shape[1])

        key = (method, arr.shape, tuple(map(str, names)), self._fingerprint(arr))
        if key in self._cache:
            self.cache_hits_ += 1
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        self.cache_misses_ += 1

        if method == "mutual_info":
            matrix = self._mutual_info_matrix(arr)
        else:
            matrix = self._correlation_matrix(arr, method)
        frame = pd.DataFrame(matrix, index=names, columns=names)
        self._cache[key] = frame
        while len(self._cache) > max(self.cache_size, 0):
            self._cache.popitem(last=False)
        return frame.copy()

    def clear_cache(self) -> None:
        self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()  # recomputable; can be large for wide data
        return state

    def _effective_method(self, n_features: int) -> str:
        if self.method == "mutual_info" and n_features > self.mi_max_features:
            warnings.warn(
                f"mutual_info dependence is O(n_features^2) and too slow for {n_features} features "
                f"(> mi_max_features={self.mi_max_features}); falling back to Spearman correlation.",
                UserWarning,
                stacklevel=4,
            )
            return "spearman"
        return self.method

    @staticmethod
    def _fingerprint(arr: np.ndarray) -> str:
        return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=16).hexdigest()

    def _subsample(self, arr: np.ndarray, limit: int) -> np.ndarray:
        if limit and arr.shape[0] > limit:
            rng = np.random.default_rng(self.random_state)
            return arr[np.sort(rng.choice(arr.shape[0], limit, replace=False))]
        return arr

    def _correlation_matrix(self, arr: np.ndarray, method: str) -> np.ndarray:
        arr = self._subsample(np.where(np.isfinite(arr), arr, np.nan), self.max_samples)
        if np.isnan(arr).any():
            # Pairwise-complete observations; pandas handles the NaN bookkeeping.
            corr = pd.DataFrame(arr).corr(method=method).to_numpy()
        else:
            data = stats.rankdata(arr, axis=0) if method == "spearman" else arr
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(data, rowvar=False)
            corr = np.atleast_2d(corr)
        # Constant columns give NaN correlations: treat them as independent of everything.
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def _mutual_info_matrix(self, arr: np.ndarray) -> np.ndarray:
        from sklearn.feature_selection import mutual_info_regression

        arr = self._subsample(np.where(np.isfinite(arr), arr, np.nan), self.mi_max_samples)
        # The kNN MI estimator cannot handle NaN: impute column medians.
        medians = np.nanmedian(np.where(np.isnan(arr).all(axis=0), 0.0, arr), axis=0)
        arr = np.where(np.isnan(arr), medians, arr)
        n_features = arr.shape[1]
        mi = np.zeros((n_features, n_features))
        for j in range(n_features):
            if np.ptp(arr[:, j]) == 0:
                continue  # constant target column: MI with it is 0
            mi[:, j] = mutual_info_regression(arr, arr[:, j], random_state=self.random_state)
        mi = np.maximum((mi + mi.T) / 2.0, 0.0)  # kNN estimates are not exactly symmetric
        dependence = np.sqrt(1.0 - np.exp(-2.0 * mi))  # Linfoot's informational coefficient of correlation
        np.fill_diagonal(dependence, 1.0)
        return dependence

    # ------------------------------------------------------------------ clustering

    def cluster(self, correlation: pd.DataFrame) -> Dict[int, List[Hashable]]:
        """Cluster features from a dependence matrix. Returns cluster_id -> features."""
        names = list(correlation.index)
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {0: names}
        distance = 1.0 - np.abs(correlation.to_numpy(dtype=np.float64))
        distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        tree = hierarchy.linkage(squareform(distance, checks=False), method=self.linkage)
        labels = hierarchy.fcluster(tree, t=self.distance_threshold, criterion="distance")
        # Renumber 0..K-1 by first appearance in column order (deterministic ids).
        mapping: Dict[int, int] = {}
        clusters: Dict[int, List[Hashable]] = {}
        for name, label in zip(names, labels):
            cid = mapping.setdefault(int(label), len(mapping))
            clusters.setdefault(cid, []).append(name)
        return clusters

    # ------------------------------------------------------------------ stability

    @staticmethod
    def stability(cluster_importances: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Per-cluster rank stability across snapshots.

        ``cluster_importances`` has shape ``(n_snapshots, n_clusters)``. For
        every pair of consecutive snapshots and every pair of clusters
        ``(c, d)``, the pair is *concordant* (+1) if their order is the same in
        both snapshots (ties included), *discordant* (-1) if it flips, and 0 if
        a tie appears or disappears. A cluster's score is its mean over the
        other clusters and over consecutive snapshot pairs. This is Kendall's
        tau decomposed per item: the scores average to the snapshots' Kendall
        tau-a.

        ``+1`` means the cluster kept its position relative to every other
        cluster throughout, i.e. a reliable signal. Values near 0 or below mean
        its importance moves around and should not be trusted.

        Returns ``(per_cluster_scores, overall)``, where ``overall`` is the
        mean Kendall's tau-b between consecutive snapshots (None if undefined).
        """
        values = np.asarray(cluster_importances, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("Need an (n_snapshots >= 2, n_clusters) array.")
        n_snap, k = values.shape
        if k == 1:
            return np.ones(1), None
        scores = np.zeros(k)
        taus = []
        for t in range(n_snap - 1):
            a, b = values[t], values[t + 1]
            sa = np.sign(a[:, None] - a[None, :])
            sb = np.sign(b[:, None] - b[None, :])
            agree = np.where(sa == sb, 1.0, np.where(sa * sb < 0, -1.0, 0.0))
            np.fill_diagonal(agree, 0.0)
            scores += agree.sum(axis=1) / (k - 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tau = stats.kendalltau(a, b).statistic
            if np.isfinite(tau):
                taus.append(float(tau))
        return scores / (n_snap - 1), (float(np.mean(taus)) if taus else None)

    # ------------------------------------------------------------------ main entry point

    def compute(
        self,
        model,
        X,
        y=None,
        snapshots: Optional[Sequence[Union[Dict[Hashable, float], Sequence[float], np.ndarray]]] = None,
        correlation: Optional[pd.DataFrame] = None,
    ) -> ClusteredImportanceReport:
        """Build a :class:`ClusteredImportanceReport`.

        Parameters
        ----------
        model : lightgbm.Booster, LGBMRegressor/LGBMClassifier, or IncrementalLGBM
            Anything that is a Booster or exposes a fitted ``booster_``.
        X : DataFrame or array
            Data for the dependence matrix (and permutation importance). If it
            is a DataFrame whose columns match the booster's feature names, it
            is aligned by name. Otherwise columns are taken in order.
        y : array-like, optional
            If given, clustered permutation importance (joint permutation of
            each cluster's columns) is added to the report.
        snapshots : sequence, optional
            Feature importance snapshots, oldest first. Each is a dict
            feature -> importance or an array in column order. Two or more
            enable stability scores.
        correlation : DataFrame, optional
            Precomputed dependence matrix to use instead of computing one.
        """
        booster = _extract_booster(model)
        arr, names = as_2d_float(X)
        booster_names = booster.feature_name()
        if arr.shape[1] != len(booster_names):
            raise ValueError(f"X has {arr.shape[1]} features but the model has {len(booster_names)}.")
        if names is not None and set(map(str, names)) == set(booster_names):
            position = {str(n): i for i, n in enumerate(names)}
            order = [position[b] for b in booster_names]
            arr = arr[:, order]
            names = [names[i] for i in order]
        feature_names: List[Hashable] = list(names) if names is not None else list(booster_names)

        if correlation is None:
            correlation = self.correlation(arr, feature_names)
        else:
            correlation = correlation.loc[feature_names, feature_names]
        clusters = self.cluster(correlation)

        raw = booster.feature_importance(importance_type=self.importance_type).astype(np.float64)
        raw_by_name = dict(zip(feature_names, raw))
        cluster_importance: Dict[int, float] = {}
        feature_importance: Dict[Hashable, Tuple[int, float]] = {}
        for cid, members in clusters.items():
            total = float(sum(raw_by_name[f] for f in members))
            cluster_importance[cid] = total
            for f in members:
                # A cluster the model never uses gets uniform shares, so shares
                # always sum to 1 within a cluster.
                share = raw_by_name[f] / total if total > 0 else 1.0 / len(members)
                feature_importance[f] = (cid, float(share))

        stability_scores, overall, n_snap = None, None, 0
        if snapshots is not None and len(snapshots) >= 2:
            matrix = self._snapshot_matrix(snapshots, feature_names)
            index = {f: i for i, f in enumerate(feature_names)}
            per_cluster = np.column_stack(
                [matrix[:, [index[f] for f in members]].sum(axis=1) for members in clusters.values()]
            )
            scores, overall = self.stability(per_cluster)
            stability_scores = {cid: float(s) for cid, s in zip(clusters, scores)}
            n_snap = len(snapshots)

        perm, perm_metric = None, None
        if y is not None:
            perm, perm_metric = self._permutation_importance(booster, arr, as_1d(y), clusters, feature_names)

        return ClusteredImportanceReport(
            clusters=clusters,
            cluster_importance=cluster_importance,
            feature_importance=feature_importance,
            stability_scores=stability_scores,
            correlation_matrix=correlation,
            method=self._effective_method_name(correlation),
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            importance_type=self.importance_type,
            raw_feature_importance={f: float(v) for f, v in raw_by_name.items()},
            cluster_permutation_importance=perm,
            permutation_metric=perm_metric,
            overall_stability=overall,
            n_snapshots=n_snap,
        )

    def _effective_method_name(self, correlation: pd.DataFrame) -> str:
        if self.method == "mutual_info" and correlation.shape[0] > self.mi_max_features:
            return "spearman"
        return self.method

    @staticmethod
    def _snapshot_matrix(snapshots, feature_names: List[Hashable]) -> np.ndarray:
        rows = []
        for snap in snapshots:
            if isinstance(snap, dict):
                rows.append([float(snap.get(f, 0.0)) for f in feature_names])
            else:
                row = np.asarray(snap, dtype=np.float64).ravel()
                if row.shape[0] != len(feature_names):
                    raise ValueError("Array snapshots must have one entry per feature, in column order.")
                rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def _permutation_importance(self, booster, arr, y, clusters, feature_names):
        """Clustered MDA: loss increase when a cluster's columns are permuted *jointly*.

        One row permutation is applied to all of a cluster's columns, which
        keeps the within-cluster dependence intact and breaks only the
        cluster's link to the target. Correlated substitutes can therefore
        not cover for each other, which is the substitution effect that
        biases single-feature permutation importance.
        """
        if y.shape[0] != arr.shape[0]:
            raise ValueError("X and y have different numbers of rows.")
        rng = np.random.default_rng(self.random_state)
        if self.permutation_max_samples and arr.shape[0] > self.permutation_max_samples:
            rows = np.sort(rng.choice(arr.shape[0], self.permutation_max_samples, replace=False))
            arr, y = arr[rows], y[rows]
        objective, options = booster_objective(booster)
        metric, loss = loss_function(objective, options)
        y = y.astype(np.float64)
        base = loss(y, booster.predict(arr, raw_score=True, num_iteration=-1))
        index = {f: i for i, f in enumerate(feature_names)}
        out = {}
        for cid, members in clusters.items():
            cols = [index[f] for f in members]
            deltas = []
            for _ in range(max(self.n_repeats, 1)):
                shuffled = arr.copy()
                shuffled[:, cols] = arr[rng.permutation(arr.shape[0])][:, cols]
                deltas.append(loss(y, booster.predict(shuffled, raw_score=True, num_iteration=-1)) - base)
            out[cid] = float(np.mean(deltas))
        return out, (metric if isinstance(metric, str) else "custom")
