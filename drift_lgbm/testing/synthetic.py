"""Synthetic financial feature data with a known correlation and regime structure.

The generator mimics the properties that break naive incremental training and
naive feature importance:

* **Correlated feature families**: several observable features per latent
  factor, like momentum over different look-backs or several volatility
  estimators. They are noisy copies of one signal, with varying noise, so some
  copies are cleaner than others.
* **Stand-alone informative features**, each its own signal.
* **Pure noise features**.
* **An optional regime change**: from a given row on, every latent signal's
  mean shifts and its volatility rises. Noise features stay put.

Everything the generator did is returned in a metadata dict, so tests can check
models against the ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["generate_financial_data"]

# Feature families (one per correlated group), in the order groups are created.
_FAMILIES: List[Tuple[str, List[str]]] = [
    ("momentum", ["momentum_5d", "momentum_10d", "momentum_21d", "momentum_63d", "momentum_126d", "momentum_252d"]),
    (
        "volatility",
        ["vol_realized_10d", "vol_realized_21d", "vol_implied_30d", "vol_parkinson_21d", "vol_garman_klass_21d", "vol_implied_60d"],
    ),
    ("value", ["value_pe", "value_pb", "value_ev_ebitda", "value_fcf_yield", "value_div_yield", "value_earnings_yield"]),
    (
        "liquidity",
        ["volume_adv_20d", "volume_turnover_21d", "volume_amihud_21d", "volume_rel_5d", "volume_dollar_63d", "volume_obv_slope"],
    ),
    ("quality", ["quality_roe", "quality_roa", "quality_gross_margin", "quality_accruals", "quality_leverage", "quality_eps_stability"]),
    ("sentiment", ["sentiment_news_1d", "sentiment_news_5d", "sentiment_social_1d", "sentiment_revisions", "sentiment_short_int", "sentiment_put_call"]),
    ("carry", ["carry_fx_3m", "carry_rates_1y", "carry_roll_yield", "carry_basis_3m", "carry_div_futures", "carry_term_slope"]),
    ("size", ["size_log_mcap", "size_log_ev", "size_log_sales", "size_log_assets", "size_log_float", "size_log_employees"]),
    ("macro", ["macro_curve_2s10s", "macro_credit_spread", "macro_breakeven_5y", "macro_pmi_surprise", "macro_usd_index", "macro_oil_mom"]),
    ("reversal", ["reversal_1d", "reversal_5d", "reversal_intraday", "reversal_overnight", "reversal_week", "reversal_gap"]),
]

_STANDALONE = [
    "earnings_surprise", "beta_252d", "skew_realized_63d", "rsi_14d", "macd_signal", "bollinger_width_20d",
    "idio_vol_63d", "max_return_21d", "insider_buy_ratio", "seasonality_month", "high_low_range_5d",
    "dist_52w_high", "analyst_dispersion", "options_iv_skew", "etf_flow_5d",
]


def _family(g: int, features_per_group: int) -> Tuple[str, List[str]]:
    if g < len(_FAMILIES):
        name, names = _FAMILIES[g]
    else:
        name, names = f"factor{g}", []
    names = list(names[:features_per_group])
    names += [f"{name}_{k}" for k in range(len(names), features_per_group)]
    return name, names


def generate_financial_data(
    n_samples: int = 10000,
    n_informative: int = 10,
    n_correlated_groups: int = 3,
    features_per_group: int = 5,
    n_noise: int = 10,
    regime_change_at: Optional[int] = None,
    regime_shift_magnitude: float = 2.0,
    target_noise: float = 0.1,
    random_state: Optional[int] = 42,
    n_informative_groups: Optional[int] = None,
    group_noise_range: Tuple[float, float] = (0.2, 0.8),
    regime_concept_shift: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Generate ``(X, y, metadata)`` with a known correlation and regime structure.

    Parameters
    ----------
    n_samples : int
        Rows, in time order.
    n_informative : int
        Stand-alone informative features, each its own latent signal that
        enters the target.
    n_correlated_groups : int
        Latent factors observed through several correlated features each
        (momentum, volatility, value, ... then ``factor<k>``).
    features_per_group : int
        Observable features per factor: ``feature = signal + sigma * noise``,
        with ``sigma`` spread evenly over ``group_noise_range`` and shuffled, so
        the cleanest copy is not always the first one.
    n_noise : int
        Pure-noise features, unrelated to the target and unaffected by the regime change.
    regime_change_at : int, optional
        Row where the regime changes. From this row on, every latent signal
        ``s`` becomes ``s * (1 + magnitude / 4) + direction * magnitude`` with a
        random ``direction`` in ``{-1, +1}`` per signal: a mean shift of
        ``magnitude`` pre-regime standard deviations, plus higher volatility.
    regime_shift_magnitude : float
        Size of the shift (default 2.0).
    target_noise : float
        Noise-to-signal ratio of the target: the noise standard deviation is
        ``target_noise`` times the (pre-regime) standard deviation of the
        signal part.
    random_state : int, optional
        Seed for full reproducibility.
    n_informative_groups : int, optional
        Only the first ``n_informative_groups`` factors enter the target. The
        rest are correlated but *uninformative*. Default: all factors.
    group_noise_range : (float, float)
        Range of the observation-noise ``sigma`` within a factor. With the
        default, copies correlate with each other at ~0.6-0.95.
    regime_concept_shift : float, default 0.0
        Optional *concept* drift: the fraction of target-relevant signals
        whose coefficient flips sign at the regime change, like a factor
        premium reversing (e.g. a momentum crash). With the default 0 the
        regime change is a pure covariate shift, where ``y | signals`` is
        unchanged. Requires ``regime_change_at``.

    Returns
    -------
    X : pandas.DataFrame
        Factor features first (family by family), then the stand-alone
        informative features, then the noise features.
    y : pandas.Series
        Linear combination of the latent signals, plus noise, named ``target``.
    metadata : dict
        ``groups`` (factor -> features), ``informative_groups``,
        ``uninformative_groups``, ``informative_features`` (stand-alone),
        ``noise_features``, ``coefficients`` (signal -> weight),
        ``coefficients_after_regime`` and ``flipped_signals`` (concept drift),
        ``feature_noise_levels``, ``feature_to_group``, ``shifted_features``,
        ``regime_change_at``, ``regime_shift`` (signal -> mean shift and scale),
        ``target_noise_std`` and ``random_state``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    for name, value in [
        ("n_informative", n_informative),
        ("n_correlated_groups", n_correlated_groups),
        ("n_noise", n_noise),
    ]:
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
    if n_correlated_groups and features_per_group < 1:
        raise ValueError("features_per_group must be >= 1.")
    if regime_change_at is not None and not 0 < regime_change_at < n_samples:
        raise ValueError("regime_change_at must be strictly between 0 and n_samples.")
    if target_noise < 0:
        raise ValueError("target_noise must be >= 0.")
    if n_informative_groups is None:
        n_informative_groups = n_correlated_groups
    if not 0 <= n_informative_groups <= n_correlated_groups:
        raise ValueError("n_informative_groups must be between 0 and n_correlated_groups.")
    lo, hi = group_noise_range
    if not 0 <= lo <= hi:
        raise ValueError("group_noise_range must satisfy 0 <= low <= high.")
    if not 0.0 <= regime_concept_shift <= 1.0:
        raise ValueError("regime_concept_shift must be in [0, 1].")
    if regime_concept_shift and regime_change_at is None:
        raise ValueError("regime_concept_shift requires regime_change_at.")

    rng = np.random.default_rng(random_state)
    n_signals = n_correlated_groups + n_informative
    families = [_family(g, features_per_group) for g in range(n_correlated_groups)]
    standalone = [
        _STANDALONE[i] if i < len(_STANDALONE) else f"alpha_signal_{i}" for i in range(n_informative)
    ]
    signal_names = [name for name, _ in families] + standalone

    # Latent signals (unit variance pre-regime), then the optional regime change.
    signals = rng.standard_normal((n_samples, n_signals))
    regime_shift = None
    if regime_change_at is not None and n_signals:
        direction = rng.choice([-1.0, 1.0], size=n_signals)
        mean_shift = regime_shift_magnitude * direction
        scale = 1.0 + regime_shift_magnitude / 4.0
        signals[regime_change_at:] = signals[regime_change_at:] * scale + mean_shift
        regime_shift = {
            name: {"mean_shift": float(m), "scale": float(scale)} for name, m in zip(signal_names, mean_shift)
        }

    # Observable features.
    columns: Dict[str, np.ndarray] = {}
    groups: Dict[str, List[str]] = {}
    noise_levels: Dict[str, float] = {}
    base_levels = np.linspace(lo, hi, features_per_group) if features_per_group > 1 else np.array([lo])
    for g, (family, names) in enumerate(families):
        sigmas = rng.permutation(base_levels)
        for name, sigma in zip(names, sigmas):
            columns[name] = signals[:, g] + sigma * rng.standard_normal(n_samples)
            noise_levels[name] = float(sigma)
        groups[family] = names
    for i, name in enumerate(standalone):
        columns[name] = signals[:, n_correlated_groups + i].copy()
    noise_features = [f"noise_{i:02d}" for i in range(n_noise)]
    for name in noise_features:
        columns[name] = rng.standard_normal(n_samples)
    X = pd.DataFrame(columns)

    # Target: linear in the latent signals.
    group_coef = rng.uniform(0.5, 1.5, n_correlated_groups) * rng.choice([-1.0, 1.0], n_correlated_groups)
    group_coef[n_informative_groups:] = 0.0
    standalone_coef = rng.uniform(0.2, 1.0, n_informative) * rng.choice([-1.0, 1.0], n_informative)
    coef = np.concatenate([group_coef, standalone_coef])
    signal_part = signals @ coef if n_signals else np.zeros(n_samples)
    noise_std = float(target_noise * np.sqrt(np.sum(coef**2)))  # pre-regime signal std
    target_noise_draw = noise_std * rng.standard_normal(n_samples)

    # Optional concept drift. These draws happen last, so the default output is
    # identical to the pure-covariate-shift generator.
    coef_after = coef.copy()
    flipped: List[str] = []
    if regime_concept_shift > 0 and n_signals:
        active = np.flatnonzero(coef != 0.0)
        n_flip = int(round(regime_concept_shift * len(active)))
        chosen = np.sort(rng.choice(active, size=n_flip, replace=False)) if n_flip else np.array([], dtype=int)
        coef_after[chosen] *= -1.0
        flipped = [signal_names[i] for i in chosen]
        signal_part = signal_part.copy()
        signal_part[regime_change_at:] = signals[regime_change_at:] @ coef_after
    y = pd.Series(signal_part + target_noise_draw, name="target")

    informative_groups = [family for family, c in zip(groups, group_coef) if c != 0.0]
    feature_to_group = {f: family for family, names in groups.items() for f in names}
    feature_to_group.update({f: None for f in standalone + noise_features})
    shifted = [] if regime_shift is None else [f for names in groups.values() for f in names] + standalone

    metadata = {
        "groups": groups,
        "informative_groups": informative_groups,
        "uninformative_groups": [family for family in groups if family not in informative_groups],
        "informative_features": standalone,
        "noise_features": noise_features,
        "coefficients": dict(zip(signal_names, coef.astype(float))),
        "coefficients_after_regime": dict(zip(signal_names, coef_after.astype(float))),
        "flipped_signals": flipped,
        "feature_noise_levels": noise_levels,
        "feature_to_group": feature_to_group,
        "shifted_features": shifted,
        "regime_change_at": regime_change_at,
        "regime_shift": regime_shift,
        "target_noise_std": noise_std,
        "random_state": random_state,
    }
    return X, y, metadata
