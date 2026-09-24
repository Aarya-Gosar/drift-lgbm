"""Per-feature distribution tracking and drift detection.

The tracker keeps its *own* histograms, separate from LightGBM's internal bin
mappers. Bin edges are quantiles of a reference batch (the first batch, or the
batch passed to :meth:`DistributionTracker.reset`), so every reference bin holds
roughly ``1 / n_bins`` of the mass.

Drift is detected with a two-sample Kolmogorov-Smirnov test computed on that
quantile grid::

    D = max_k max(|F_ref(e_k) - F_new(e_k)|, |F_ref(e_k-) - F_new(e_k-)|)

where ``e_k`` are the reference bin edges and ``F(e-)`` is the left limit of
the empirical CDF. Evaluating the CDFs only at the reference quantiles means no
raw reference data is kept (the reference distribution is a fixed-size digest,
which keeps the tracker small and cheap to pickle). The cost is a resolution of
about ``1 / n_bins`` in ``D``. Because the grid statistic can only
*under*-estimate the exact KS statistic, the p-values are conservative: slightly
fewer false alarms, slightly less power. Including the left limits means shifts
outside the reference range, and changes to constant features, are still caught
exactly.

NaN and +/-inf values are treated as missing and ignored in every distribution
computation. Missing counts are tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special

from ._utils import align_columns, as_2d_float

__all__ = ["DistributionTracker", "DriftReport", "FeatureStats"]


@dataclass
class DriftReport:
    """Result of comparing one batch against the tracker's reference distribution.

    All per-feature dictionaries are keyed by feature name (the DataFrame column
    name, or the integer column index for array input).

    Attributes
    ----------
    batch_index : int
        0-based index of the batch among all batches the tracker has seen.
    n_samples : int
        Number of rows in the batch.
    feature_names : list
        Features in column order.
    p_value_threshold : float
        The KS p-value threshold used for this batch. It can be higher than the
        configured ``drift_threshold`` for small batches (see
        :meth:`DistributionTracker.effective_threshold`).
    drift_fraction_threshold : float
        Fraction of drifted features above which a rebin is recommended.
    ks_statistic : dict
        KS distance in ``[0, 1]`` between the reference and the batch.
    p_value : dict
        KS test p-value (1.0 if the feature could not be tested, e.g. all-NaN).
    mean_shift : dict
        ``(mean_batch - mean_ref) / std_ref``: the shift in reference standard
        deviations (``+/-inf`` for a constant reference feature whose mean moved).
    std_ratio : dict
        ``std_batch / std_ref``.
    out_of_range_fraction : dict
        Fraction of the batch's non-missing values outside ``[ref_min, ref_max]``.
        These are the values that frozen bins would cram into the edge bins.
    drifted_features : list
        Features flagged as drifted, most drifted (largest KS distance) first.
    is_reference : bool
        True if this batch *established* the reference, in which case nothing
        can have drifted.
    """

    batch_index: int
    n_samples: int
    feature_names: List[Hashable]
    p_value_threshold: float
    drift_fraction_threshold: float
    ks_statistic: Dict[Hashable, float]
    p_value: Dict[Hashable, float]
    mean_shift: Dict[Hashable, float]
    std_ratio: Dict[Hashable, float]
    out_of_range_fraction: Dict[Hashable, float]
    drifted_features: List[Hashable]
    is_reference: bool = False

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_drifted(self) -> int:
        return len(self.drifted_features)

    @property
    def fraction_drifted(self) -> float:
        return self.n_drifted / self.n_features if self.n_features else 0.0

    @property
    def rebin_recommended(self) -> bool:
        """True if strictly more than ``drift_fraction_threshold`` of features drifted."""
        return (not self.is_reference) and self.fraction_drifted > self.drift_fraction_threshold

    def to_frame(self) -> pd.DataFrame:
        """Per-feature drift statistics as a DataFrame, sorted by KS distance."""
        drifted = set(self.drifted_features)
        frame = pd.DataFrame(
            {
                "ks_statistic": [self.ks_statistic[f] for f in self.feature_names],
                "p_value": [self.p_value[f] for f in self.feature_names],
                "mean_shift": [self.mean_shift[f] for f in self.feature_names],
                "std_ratio": [self.std_ratio[f] for f in self.feature_names],
                "out_of_range_fraction": [self.out_of_range_fraction[f] for f in self.feature_names],
                "drifted": [f in drifted for f in self.feature_names],
            },
            index=pd.Index(self.feature_names, name="feature"),
        )
        return frame.sort_values("ks_statistic", ascending=False)

    def to_string(self, max_features: int = 20) -> str:
        if self.is_reference:
            return (
                f"DriftReport(batch {self.batch_index}): reference batch established "
                f"({self.n_samples} rows, {self.n_features} features)."
            )
        lines = [
            f"DriftReport(batch {self.batch_index}): {self.n_drifted}/{self.n_features} features "
            f"drifted ({self.fraction_drifted:.0%}, rebin threshold "
            f">{self.drift_fraction_threshold:.0%}) at p < {self.p_value_threshold:.3g}"
            f" -> rebin {'RECOMMENDED' if self.rebin_recommended else 'not needed'}",
        ]
        if self.drifted_features:
            lines.append(f"  {'feature':<28}{'KS':>8}{'p-value':>12}{'mean shift':>12}{'out-of-range':>14}")
            for f in self.drifted_features[:max_features]:
                lines.append(
                    f"  {str(f)[:27]:<28}{self.ks_statistic[f]:>8.3f}{self.p_value[f]:>12.2e}"
                    f"{self.mean_shift[f]:>+12.2f}{self.out_of_range_fraction[f]:>14.1%}"
                )
            if self.n_drifted > max_features:
                lines.append(f"  ... and {self.n_drifted - max_features} more")
        return "\n".join(lines)

    def summary(self, max_features: int = 20) -> None:
        """Print a human-readable summary."""
        print(self.to_string(max_features=max_features))

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class FeatureStats:
    """Tracked statistics for a single feature (see :meth:`DistributionTracker.feature_stats`).

    ``reference_*`` fields describe the reference batch. ``running_*`` fields
    cover every batch seen since the reference was established, the reference
    batch included.
    """

    name: Hashable
    reference_count: int
    reference_missing: int
    reference_min: float
    reference_max: float
    reference_mean: float
    reference_std: float
    running_count: int
    running_missing: int
    running_min: float
    running_max: float
    running_underflow: int
    running_overflow: int
    n_bins: int

    @property
    def running_out_of_range_fraction(self) -> float:
        if self.running_count == 0:
            return 0.0
        return (self.running_underflow + self.running_overflow) / self.running_count


def _sorted_quantiles(sorted_values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantiles of already-sorted data (numpy's default linear interpolation)."""
    n = sorted_values.shape[0]
    if n == 1:
        return np.full(levels.shape, sorted_values[0], dtype=np.float64)
    pos = levels * (n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _ks_pvalue(d: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """Two-sample KS p-value at effective sample size ``n_eff = n*m/(n+m)``.

    Uses the asymptotic Kolmogorov distribution with Stephens' small-sample
    correction, ``Q_KS((sqrt(n) + 0.12 + 0.11/sqrt(n)) * D)``. Pragmatic choice:
    ``scipy.stats.kstwo.sf`` is ~10ms per feature (seconds for a wide batch),
    while this is effectively free. It stays within ~10% of the exact
    one-sample value for ``n >= 25`` at p in [0.001, 0.1], and errs on the
    conservative side (larger p) below that.
    """
    root = np.sqrt(np.maximum(n_eff, 1.0))
    p = special.kolmogorov((root + 0.12 + 0.11 / root) * d)
    return np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)


def _bin_counts(left: np.ndarray, right: np.ndarray, n: int) -> Tuple[np.ndarray, int, int]:
    """Histogram counts over bins defined by unique edges.

    ``left[k]`` / ``right[k]`` are the number of values ``< e_k`` / ``<= e_k``.
    Bins are ``[e_0, e_1), ..., [e_{m-1}, e_m]`` (the last bin is closed). A
    single edge (a constant reference feature) gives one degenerate bin
    ``[e_0, e_0]``. Returns ``(counts, underflow, overflow)``.
    """
    if left.shape[0] == 1:
        counts = np.array([right[0] - left[0]], dtype=np.int64)
    else:
        counts = (left[1:] - left[:-1]).astype(np.int64)
        counts[-1] = right[-1] - left[-2]
    return counts, int(left[0]), int(n - right[-1])


class DistributionTracker:
    """Track per-feature distributions across batches and detect drift.

    The first batch passed to :meth:`update` (or :meth:`fit` / :meth:`reset`)
    becomes the *reference*: its quantiles define ``n_bins`` bin edges per
    feature, and the reference histogram and summary statistics are stored.
    Each later batch is KS-tested against the reference (see module docstring)
    and its counts are added to a running histogram over the reference bins.

    Parameters
    ----------
    n_bins : int, default=256
        Number of quantile bins per feature. Ties, e.g. from discrete features,
        collapse duplicate edges, so a feature can end up with fewer bins.
    drift_threshold : float, default=0.01
        A feature is flagged as drifted when its KS p-value is below this.
    drift_fraction : float, default=0.3
        :meth:`should_rebin` returns True when *strictly more* than this
        fraction of features drifted in the latest batch.
    small_batch_size : int, default=500
        Batches with fewer rows than this get a looser p-value threshold. The
        KS test has little power on small samples, so real drift would go
        unflagged at the nominal threshold. The threshold is scaled by
        ``small_batch_size / n_rows`` and capped at
        ``max_small_batch_threshold``. Set to 0 to disable.
    max_small_batch_threshold : float, default=0.1
        Upper bound for the loosened small-batch threshold. This bounds the
        extra false alarms the loosening can cause.
    min_ks_statistic : float, default=0.0
        Optional effect-size floor: a feature only counts as drifted if its KS
        distance is at least this large. On very large batches the KS test flags
        practically irrelevant shifts; e.g. 0.05 ignores shifts that move the
        CDF by less than 5 points. Disabled (0.0) by default.

    Notes
    -----
    The tracker is a plain Python object holding numpy arrays, so it can be
    pickled. It is not thread-safe.
    """

    def __init__(
        self,
        n_bins: int = 256,
        drift_threshold: float = 0.01,
        drift_fraction: float = 0.3,
        small_batch_size: int = 500,
        max_small_batch_threshold: float = 0.1,
        min_ks_statistic: float = 0.0,
    ):
        if int(n_bins) < 2:
            raise ValueError("n_bins must be >= 2.")
        if not 0.0 < drift_threshold < 1.0:
            raise ValueError("drift_threshold must be in (0, 1).")
        if not 0.0 <= drift_fraction < 1.0:
            raise ValueError("drift_fraction must be in [0, 1).")
        if small_batch_size < 0:
            raise ValueError("small_batch_size must be >= 0.")
        if not 0.0 <= min_ks_statistic <= 1.0:
            raise ValueError("min_ks_statistic must be in [0, 1].")
        self.n_bins = int(n_bins)
        self.drift_threshold = float(drift_threshold)
        self.drift_fraction = float(drift_fraction)
        self.small_batch_size = int(small_batch_size)
        self.max_small_batch_threshold = float(max_small_batch_threshold)
        self.min_ks_statistic = float(min_ks_statistic)

        self.feature_names_: Optional[List[Hashable]] = None
        self.n_batches_seen_ = 0
        self.last_report_: Optional[DriftReport] = None

    # ------------------------------------------------------------------ public API

    @property
    def is_fitted(self) -> bool:
        return self.feature_names_ is not None

    @property
    def n_features_(self) -> int:
        self._check_fitted()
        return len(self.feature_names_)

    def fit(self, X) -> "DistributionTracker":
        """Establish the reference distribution from ``X``. Same as :meth:`reset`."""
        return self.reset(X)

    def reset(self, X) -> "DistributionTracker":
        """(Re-)establish the reference distribution from ``X``.

        This discards the previous reference and running histograms. Call it
        after a rebin so that later drift is measured against the distribution
        the current model was rebuilt on.
        """
        arr, names = as_2d_float(X)
        if arr.shape[0] == 0:
            raise ValueError("Cannot establish a reference from an empty batch.")
        if self.is_fitted:
            arr = align_columns(arr, names, self.feature_names_, self._has_names)
        else:
            self._has_names = names is not None
            self.feature_names_ = list(names) if names is not None else list(range(arr.shape[1]))
        self._establish_reference(arr)
        return self

    def update(self, X) -> DriftReport:
        """Ingest a batch and report drift against the reference.

        If the tracker has no reference yet, ``X`` becomes the reference and
        the returned report has ``is_reference=True`` and no drifted features.
        """
        if not self.is_fitted:
            self.reset(X)
            report = self._reference_report(self._last_reference_rows)
        else:
            arr, names = as_2d_float(X)
            if arr.shape[0] == 0:
                raise ValueError("Cannot update with an empty batch.")
            arr = align_columns(arr, names, self.feature_names_, self._has_names)
            report = self._compare_and_accumulate(arr)
        self.last_report_ = report
        self.n_batches_seen_ += 1
        return report

    def should_rebin(self, report: Optional[DriftReport] = None) -> bool:
        """True if strictly more than ``drift_fraction`` of features drifted.

        Uses the given report, or the latest one from :meth:`update`. Returns
        False before any batch has been compared against the reference.
        """
        report = report if report is not None else self.last_report_
        if report is None:
            return False
        return report.rebin_recommended

    def effective_threshold(self, n_samples: int) -> float:
        """p-value threshold used for a batch with ``n_samples`` rows.

        Batches smaller than ``small_batch_size`` get a proportionally looser
        threshold, since the KS test is underpowered on them. The loosened
        value is capped at ``max_small_batch_threshold`` and is never tighter
        than ``drift_threshold``.
        """
        thr = self.drift_threshold
        if self.small_batch_size and 0 < n_samples < self.small_batch_size:
            loosened = thr * self.small_batch_size / n_samples
            thr = max(thr, min(loosened, self.max_small_batch_threshold))
        return thr

    def feature_stats(self, feature: Hashable) -> FeatureStats:
        """Tracked statistics for one feature (by name, or by index for array input)."""
        j = self._feature_index(feature)
        return FeatureStats(
            name=self.feature_names_[j],
            reference_count=int(self._ref_count[j]),
            reference_missing=int(self._ref_missing[j]),
            reference_min=float(self._ref_min[j]),
            reference_max=float(self._ref_max[j]),
            reference_mean=float(self._ref_mean[j]),
            reference_std=float(self._ref_std[j]),
            running_count=int(self._run_count[j]),
            running_missing=int(self._run_missing[j]),
            running_min=float(self._run_min[j]),
            running_max=float(self._run_max[j]),
            running_underflow=int(self._run_under[j]),
            running_overflow=int(self._run_over[j]),
            n_bins=int(self._ref_hist[j].shape[0]),
        )

    def reference_quantiles(self, feature: Hashable, levels=None) -> np.ndarray:
        """Quantiles of the reference distribution, read off the stored quantile digest.

        ``levels`` defaults to the digest's own levels
        (``linspace(0, 1, n_bins + 1)``). Other levels are linearly interpolated.
        """
        j = self._feature_index(feature)
        digest = self._ref_quantiles[j]
        if levels is None:
            return digest.copy()
        levels = np.asarray(levels, dtype=np.float64)
        return np.interp(levels, self._levels, digest)

    def reference_histogram(self, feature: Hashable) -> Tuple[np.ndarray, np.ndarray]:
        """``(edges, counts)`` of the reference batch over the reference bins."""
        j = self._feature_index(feature)
        return self._edges[j].copy(), self._ref_hist[j].copy()

    def running_histogram(self, feature: Hashable) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """``(edges, counts, underflow, overflow)`` over all data since the reference was set.

        ``underflow`` / ``overflow`` count values below / above the reference
        range. These are the values that frozen, first-batch bins cannot resolve.
        """
        j = self._feature_index(feature)
        return (
            self._edges[j].copy(),
            self._run_hist[j].copy(),
            int(self._run_under[j]),
            int(self._run_over[j]),
        )

    # ------------------------------------------------------------------ internals

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("DistributionTracker has no reference yet; call update() or fit() first.")

    def _feature_index(self, feature: Hashable) -> int:
        self._check_fitted()
        try:
            return self.feature_names_.index(feature)
        except ValueError:
            raise KeyError(f"Unknown feature {feature!r}.") from None

    @staticmethod
    def _sorted_columns(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort each column with missing values moved to the end. Returns (sorted, n_valid)."""
        finite = np.isfinite(arr)
        clean = np.where(finite, arr, np.nan)
        return np.sort(clean, axis=0), finite.sum(axis=0)

    def _establish_reference(self, arr: np.ndarray) -> None:
        n_rows, n_features = arr.shape
        self._levels = np.linspace(0.0, 1.0, self.n_bins + 1)
        sorted_cols, n_valid = self._sorted_columns(arr)

        self._ref_quantiles = np.full((n_features, self.n_bins + 1), np.nan)
        self._edges: List[np.ndarray] = []
        self._ref_cdf_left: List[np.ndarray] = []
        self._ref_cdf_right: List[np.ndarray] = []
        self._ref_hist: List[np.ndarray] = []
        self._ref_count = n_valid.astype(np.int64)
        self._ref_missing = (n_rows - n_valid).astype(np.int64)
        self._ref_min = np.full(n_features, np.nan)
        self._ref_max = np.full(n_features, np.nan)
        self._ref_mean = np.full(n_features, np.nan)
        self._ref_std = np.full(n_features, np.nan)

        for j in range(n_features):
            m = int(n_valid[j])
            if m == 0:  # all-missing feature: nothing to compare against
                self._edges.append(np.empty(0))
                self._ref_cdf_left.append(np.empty(0))
                self._ref_cdf_right.append(np.empty(0))
                self._ref_hist.append(np.zeros(0, dtype=np.int64))
                continue
            v = sorted_cols[:m, j]
            quantiles = _sorted_quantiles(v, self._levels)
            edges = np.unique(quantiles)
            left = np.searchsorted(v, edges, side="left")
            right = np.searchsorted(v, edges, side="right")
            counts, _, _ = _bin_counts(left, right, m)

            self._ref_quantiles[j] = quantiles
            self._edges.append(edges)
            self._ref_cdf_left.append(left / m)
            self._ref_cdf_right.append(right / m)
            self._ref_hist.append(counts)
            self._ref_min[j], self._ref_max[j] = v[0], v[-1]
            self._ref_mean[j] = v.mean()
            self._ref_std[j] = v.std()

        # Running accumulators start with the reference batch itself.
        self._run_hist = [h.copy() for h in self._ref_hist]
        self._run_under = np.zeros(n_features, dtype=np.int64)
        self._run_over = np.zeros(n_features, dtype=np.int64)
        self._run_count = self._ref_count.copy()
        self._run_missing = self._ref_missing.copy()
        self._run_min = self._ref_min.copy()
        self._run_max = self._ref_max.copy()
        self._last_reference_rows = n_rows

    def _reference_report(self, n_rows: int) -> DriftReport:
        names = self.feature_names_
        zeros = {f: 0.0 for f in names}
        return DriftReport(
            batch_index=self.n_batches_seen_,
            n_samples=n_rows,
            feature_names=list(names),
            p_value_threshold=self.effective_threshold(n_rows),
            drift_fraction_threshold=self.drift_fraction,
            ks_statistic=dict(zeros),
            p_value={f: 1.0 for f in names},
            mean_shift=dict(zeros),
            std_ratio={f: 1.0 for f in names},
            out_of_range_fraction=dict(zeros),
            drifted_features=[],
            is_reference=True,
        )

    def _compare_and_accumulate(self, arr: np.ndarray) -> DriftReport:
        n_rows, n_features = arr.shape
        sorted_cols, n_valid = self._sorted_columns(arr)
        threshold = self.effective_threshold(n_rows)

        ks = np.zeros(n_features)
        eff_n = np.zeros(n_features)
        testable = np.zeros(n_features, dtype=bool)
        mean_shift = np.zeros(n_features)
        std_ratio = np.ones(n_features)
        out_of_range = np.zeros(n_features)

        for j in range(n_features):
            m, m_ref = int(n_valid[j]), int(self._ref_count[j])
            self._run_missing[j] += n_rows - m
            self._run_count[j] += m
            if m == 0 or m_ref == 0:
                continue  # nothing to test: all-missing batch or all-missing reference
            v = sorted_cols[:m, j]
            edges = self._edges[j]
            left = np.searchsorted(v, edges, side="left")
            right = np.searchsorted(v, edges, side="right")

            ks[j] = max(
                np.max(np.abs(left / m - self._ref_cdf_left[j])),
                np.max(np.abs(right / m - self._ref_cdf_right[j])),
            )
            eff_n[j] = m_ref * m / (m_ref + m)
            testable[j] = True

            counts, under, over = _bin_counts(left, right, m)
            out_of_range[j] = (under + over) / m
            mean, std = v.mean(), v.std()
            ref_mean, ref_std = self._ref_mean[j], self._ref_std[j]
            if ref_std > 0:
                mean_shift[j] = (mean - ref_mean) / ref_std
                std_ratio[j] = std / ref_std
            else:
                mean_shift[j] = 0.0 if mean == ref_mean else np.copysign(np.inf, mean - ref_mean)
                std_ratio[j] = 1.0 if std == 0 else np.inf

            self._run_hist[j] += counts
            self._run_under[j] += under
            self._run_over[j] += over
            self._run_min[j] = min(self._run_min[j], v[0])
            self._run_max[j] = max(self._run_max[j], v[-1])

        p_values = np.ones(n_features)
        if testable.any():
            p_values[testable] = _ks_pvalue(ks[testable], eff_n[testable])

        drifted_mask = testable & (p_values < threshold) & (ks >= self.min_ks_statistic)
        order = np.argsort(-ks, kind="stable")
        names = self.feature_names_
        drifted = [names[j] for j in order if drifted_mask[j]]

        return DriftReport(
            batch_index=self.n_batches_seen_,
            n_samples=n_rows,
            feature_names=list(names),
            p_value_threshold=threshold,
            drift_fraction_threshold=self.drift_fraction,
            ks_statistic={f: float(ks[j]) for j, f in enumerate(names)},
            p_value={f: float(p_values[j]) for j, f in enumerate(names)},
            mean_shift={f: float(mean_shift[j]) for j, f in enumerate(names)},
            std_ratio={f: float(std_ratio[j]) for j, f in enumerate(names)},
            out_of_range_fraction={f: float(out_of_range[j]) for j, f in enumerate(names)},
            drifted_features=drifted,
        )

    def __repr__(self) -> str:
        state = f"{len(self.feature_names_)} features" if self.is_fitted else "unfitted"
        return (
            f"DistributionTracker(n_bins={self.n_bins}, drift_threshold={self.drift_threshold}, "
            f"drift_fraction={self.drift_fraction}; {state}, {self.n_batches_seen_} batches seen)"
        )
