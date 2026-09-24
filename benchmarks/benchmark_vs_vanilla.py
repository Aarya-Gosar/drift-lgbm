"""Benchmark: IncrementalLGBM vs. vanilla LightGBM training strategies on drifting data.

Data: ``drift_lgbm.testing.generate_financial_data`` with a regime change halfway
through the training rows. The test set is drawn from the *latest* (post-change)
distribution.

Methods (same LightGBM hyper-parameters everywhere):

  A  vanilla LightGBM on all training rows at once (full data access)
  B  vanilla LightGBM on the first half only (the "can't fit all data" baseline)
  C  naive incremental: raw ``lgb.train(init_model=...)``, one batch at a time
  C* naive incremental with *frozen* bins (each batch's Dataset built with
     ``reference=`` the first batch). This is the failure mode that motivates
     the project. Pass --no-frozen to skip it.
  D  ``IncrementalLGBM``: fit on the first batch, then partial_fit

A and B train ``n_batches * trees_per_batch`` trees, the same capacity as C and D.

Scenarios:

  covariate  a regime change where signal means and volatilities shift, but
             y|x is unchanged
  concept    the same, plus half of the factor premia flip sign at the change

Usage::

    python benchmarks/benchmark_vs_vanilla.py                 # full: 100k rows, 200 features
    python benchmarks/benchmark_vs_vanilla.py --quick         # 20k rows, 60 features
    python benchmarks/benchmark_vs_vanilla.py --scenarios covariate --seeds 0 1 2 --output results.json

Exit status is 0 when D >= C (D's RMSE <= C's) in every run, i.e. drift-lgbm
was never worse than naive init_model. The other expected orderings are checked
and reported, but do not affect the exit status.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift_lgbm import DriftLGBMWarning, IncrementalLGBM  # noqa: E402
from drift_lgbm.testing import generate_financial_data  # noqa: E402

SCENARIOS = {"covariate": 0.0, "concept": 0.5}


def feature_layout(n_features: int) -> dict:
    """Split a feature budget into correlated families, stand-alone signals and noise."""
    n_groups = max(1, n_features // 20)
    per_group = 10
    n_informative = n_features // 5
    n_noise = n_features - n_groups * per_group - n_informative
    if n_noise < 0:
        raise ValueError("--features too small (need at least 20).")
    return dict(
        n_correlated_groups=n_groups, features_per_group=per_group, n_informative=n_informative, n_noise=n_noise
    )


def rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def run_once(args, scenario: str, seed: int) -> dict:
    n_train, n_batches = args.rows, args.batches
    batch = n_train // n_batches
    total_trees = n_batches * args.trees_per_batch
    X, y, meta = generate_financial_data(
        n_samples=n_train + args.test_rows,
        regime_change_at=n_train // 2,
        regime_concept_shift=SCENARIOS[scenario],
        random_state=seed,
        **feature_layout(args.features),
    )
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    batches = [(X_train.iloc[b * batch : (b + 1) * batch], y_train.iloc[b * batch : (b + 1) * batch]) for b in range(n_batches)]

    params = dict(objective="regression", learning_rate=args.learning_rate, num_leaves=args.num_leaves, seed=seed, verbosity=-1)
    results = {}

    model_a, t = timed(lambda: lgb.train(params, lgb.Dataset(X_train, y_train), total_trees))
    results["A"] = dict(rmse=rmse(model_a.predict(X_test), y_test), seconds=t)

    half = n_train // 2
    model_b, t = timed(lambda: lgb.train(params, lgb.Dataset(X_train.iloc[:half], y_train.iloc[:half]), total_trees))
    results["B"] = dict(rmse=rmse(model_b.predict(X_test), y_test), seconds=t)

    def naive(frozen: bool):
        booster, reference = None, None
        for Xb, yb in batches:
            if frozen and reference is not None:
                dataset = lgb.Dataset(Xb, yb, reference=reference)
            else:
                dataset = lgb.Dataset(Xb, yb, free_raw_data=False)
                if frozen:
                    reference = dataset.construct()
            booster = lgb.train(params, dataset, args.trees_per_batch, init_model=booster)
        return booster

    model_c, t = timed(lambda: naive(frozen=False))
    results["C"] = dict(rmse=rmse(model_c.predict(X_test), y_test), seconds=t)
    if not args.no_frozen:
        model_cf, t = timed(lambda: naive(frozen=True))
        results["C*"] = dict(rmse=rmse(model_cf.predict(X_test), y_test), seconds=t)

    def ours():
        model = IncrementalLGBM(
            n_estimators_per_batch=args.trees_per_batch,
            max_total_trees=max(1000, total_trees),
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            random_state=seed,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DriftLGBMWarning)
            model.fit(*batches[0])
            for Xb, yb in batches[1:]:
                model.partial_fit(Xb, yb)
        return model

    model_d, t = timed(ours)
    results["D"] = dict(rmse=rmse(model_d.predict(X_test), y_test), seconds=t)
    decisions = [
        dict(
            batch=r["batch"],
            action=r["action"],
            fraction_drifted=round(r["fraction_drifted"], 3),
            trees_retired=r["trees_retired"],
            trees_evicted=r["trees_evicted"],
            trees_added=r["trees_added"],
            trees_after=r["trees_after"],
            window_batches_used=r["window_batches_used"],
            prequential_rmse=None if r["prequential_loss"] is None else round(r["prequential_loss"], 4),
        )
        for r in model_d.history_
    ]
    return dict(
        scenario=scenario,
        seed=seed,
        rows=n_train,
        features=X.shape[1],
        batches=n_batches,
        batch_rows=batch,
        test_rows=len(y_test),
        regime_change_at=n_train // 2,
        concept_shift=SCENARIOS[scenario],
        results=results,
        decisions=decisions,
    )


def checks(run: dict) -> dict:
    r = {k: v["rmse"] for k, v in run["results"].items()}
    return {
        "D >= C (D at least as good as naive init_model)": r["D"] <= r["C"],
        "B < C < A (more data helps)": r["B"] > r["C"] > r["A"],
        "D closer to A than C is": abs(r["D"] - r["A"]) < abs(r["C"] - r["A"]),
    }


LABELS = {
    "A": "vanilla LightGBM, all rows",
    "B": "vanilla LightGBM, first half",
    "C": "naive init_model, per batch",
    "C*": "naive init_model, frozen bins",
    "D": "IncrementalLGBM (drift-lgbm)",
}


def report(run: dict) -> str:
    res = run["results"]
    lines = [
        f"### Scenario: {run['scenario']} (concept shift {run['concept_shift']}), seed {run['seed']}",
        f"{run['rows']:,} training rows x {run['features']} features, regime change at row {run['regime_change_at']:,}, "
        f"{run['batches']} batches of {run['batch_rows']:,}; test = {run['test_rows']:,} rows from the post-change distribution",
        "",
        "| Method | | Test RMSE | vs C | Train time |",
        "|---|---|---:|---:|---:|",
    ]
    c = res["C"]["rmse"]
    for key in ["A", "B", "C", "C*", "D"]:
        if key in res:
            v = res[key]
            lines.append(
                f"| {key} | {LABELS[key]} | {v['rmse']:.4f} | {100 * (v['rmse'] / c - 1):+.1f}% | {v['seconds']:.1f}s |"
            )
    lines.append("")
    for name, ok in checks(run).items():
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}")
    lines.append("")
    lines.append("D's per-batch decisions:")
    for d in run["decisions"]:
        pre = "" if d["prequential_rmse"] is None else f", prequential RMSE {d['prequential_rmse']}"
        lines.append(
            f"  batch {d['batch']}: {d['action']:<20} drifted {d['fraction_drifted']:.0%}, retired {d['trees_retired']}, "
            f"evicted {d['trees_evicted']}, added {d['trees_added']} -> {d['trees_after']} trees{pre}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=100_000, help="training rows (default 100,000)")
    parser.add_argument("--features", type=int, default=200, help="total features (default 200)")
    parser.add_argument("--batches", type=int, default=5, help="incremental batches (default 5)")
    parser.add_argument("--test-rows", type=int, default=20_000, help="post-change test rows (default 20,000)")
    parser.add_argument("--trees-per-batch", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=["covariate", "concept"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--no-frozen", action="store_true", help="skip the frozen-bins baseline C*")
    parser.add_argument("--quick", action="store_true", help="small run: 20k rows, 60 features, 5k test rows")
    parser.add_argument("--output", type=Path, help="write all results as JSON")
    args = parser.parse_args(argv)
    if args.quick:
        args.rows, args.features, args.test_rows = 20_000, 60, 5_000

    runs = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            run = run_once(args, scenario, seed)
            runs.append(run)
            print(report(run), end="\n\n", flush=True)

    if len(runs) > 1:
        print("### Summary: test RMSE relative to naive init_model (C)")
        for scenario in args.scenarios:
            ratios = [r["results"]["D"]["rmse"] / r["results"]["C"]["rmse"] for r in runs if r["scenario"] == scenario]
            print(f"- {scenario}: D/C = " + ", ".join(f"{x:.3f}" for x in ratios) + f" (mean {np.mean(ratios):.3f})")

    if args.output:
        args.output.write_text(json.dumps(runs, indent=2))
        print(f"\nResults written to {args.output}")

    ok = all(checks(r)["D >= C (D at least as good as naive init_model)"] for r in runs)
    print(f"\nNever worse than naive init_model (D >= C in every run): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
