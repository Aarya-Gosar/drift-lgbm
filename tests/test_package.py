"""Public API and the definition-of-done workflow."""

import pickle
import runpy
from pathlib import Path

import numpy as np

import drift_lgbm
from drift_lgbm.testing import generate_financial_data

ROOT = Path(__file__).resolve().parents[1]


def test_public_exports():
    from drift_lgbm import ClusteredImportanceReport, DistributionTracker, IncrementalLGBM

    assert IncrementalLGBM.__module__ == "drift_lgbm.model"
    assert ClusteredImportanceReport.__module__ == "drift_lgbm.feature_importance"
    assert DistributionTracker.__module__ == "drift_lgbm.distribution_tracker"
    for name in drift_lgbm.__all__:
        assert hasattr(drift_lgbm, name)
    assert drift_lgbm.__version__ == "0.1.0"


def test_readme_quickstart_workflow():
    """The README quickstart workflow, plus the pickle round trip."""
    X, y, _ = generate_financial_data(n_samples=9000, regime_change_at=5000, random_state=4)
    X_train, y_train = X.iloc[:4000], y.iloc[:4000]
    X_new, y_new = X.iloc[4000:7000], y.iloc[4000:7000]
    X_val, y_val = X.iloc[7000:8000], y.iloc[7000:8000]
    X_test = X.iloc[8000:]

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

    assert predictions.shape == (1000,) and np.all(np.isfinite(predictions))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X_test), predictions)


def test_quickstart_example_runs(capsys):
    runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "Clustered feature importance" in out and "pickle round-trip: OK" in out
