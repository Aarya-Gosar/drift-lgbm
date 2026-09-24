"""drift-lgbm: incremental LightGBM with drift detection, tree retirement and
correlation-aware feature importance.

Main entry points:

* :class:`IncrementalLGBM`: sklearn-style model with ``fit`` / ``partial_fit``.
* :class:`DistributionTracker`: per-feature drift detection (KS test).
* :class:`ClusteredImportanceReport`: cluster-level feature importance, as
  returned by ``IncrementalLGBM.feature_importance_report``.
"""

from .distribution_tracker import DistributionTracker, DriftReport
from .feature_importance import ClusteredImportance, ClusteredImportanceReport
from .model import DriftLGBMWarning, IncrementalLGBM
from .tree_manager import TreeManager

__version__ = "0.1.0"

__all__ = [
    "IncrementalLGBM",
    "ClusteredImportanceReport",
    "DistributionTracker",
    "ClusteredImportance",
    "DriftReport",
    "TreeManager",
    "DriftLGBMWarning",
    "__version__",
]
