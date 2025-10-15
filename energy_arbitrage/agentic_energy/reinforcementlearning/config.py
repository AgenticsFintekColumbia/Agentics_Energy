from __future__ import annotations
import os
from pathlib import Path

# Default run dirs
DEFAULT_SAVE_DIR = "runs/rllib_battery"

# Environment variables centralization (called once per process)
def apply_process_env():
    import warnings, logging
    warnings.filterwarnings("ignore")

    # Ray / warnings
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    # CPU friendliness
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
