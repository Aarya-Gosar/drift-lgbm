"""Quickstart: fit, partial_fit on new data, clustered feature importance, predict.

Run from the repository root:  python examples/quickstart.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift_lgbm.testing import generate_financial_data  # noqa: E402

# --- data: a feature panel with correlated factor families and a regime change
X, y, meta = generate_financial_data(n_samples=30_000, regime_change_at=15_000, random_state=0)
X_train, y_train = X.iloc[:10_000], y.iloc[:10_000]
X_new, y_new = X.iloc[10_000:20_000], y.iloc[10_000:20_000]  # straddles the regime change
X_val, y_val = X.iloc[20_000:25_000], y.iloc[20_000:25_000]
X_test, y_test = X.iloc[25_000:], y.iloc[25_000:]

# --- the 10-line script ------------------------------------------------------
from drift_lgbm import IncrementalLGBM  # noqa: E402

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
# ------------------------------------------------------------------------------

rmse = float(((predictions - y_test.to_numpy()) ** 2).mean() ** 0.5)
print(f"\nTest RMSE: {rmse:.4f}   trees: {model.n_trees_}")
print("Last update:", {k: model.history_[-1][k] for k in ("action", "fraction_drifted", "trees_retired", "trees_after")})

restored = pickle.loads(pickle.dumps(model))
assert (restored.predict(X_test) == predictions).all()
print("pickle round-trip: OK")
