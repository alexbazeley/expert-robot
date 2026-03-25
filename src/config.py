"""Configuration for the legislative bill passage prediction model.

Centralizes all configurable parameters: API settings, database paths,
feature engineering constants, model hyperparameters, and state-specific
legislative metadata.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- LegiScan API ---
LEGISCAN_API_KEY: str = os.getenv("LEGISCAN_API_KEY", "")
LEGISCAN_BASE_URL: str = "https://api.legiscan.com/"
LEGISCAN_MONTHLY_LIMIT: int = 30_000
LEGISCAN_CACHE_DIR: Path = RAW_DATA_DIR / "legiscan_cache"
LEGISCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Database ---
DATABASE_PATH: Path = DATA_DIR / "ohio_legislature.db"
DATABASE_URL: str = f"sqlite:///{DATABASE_PATH}"

# --- Ohio-specific ---
OHIO_STATE_ABBREVIATION: str = "OH"
OHIO_SESSIONS_OF_INTEREST: list[int] = [131, 132, 133, 134, 135, 136]
OHIO_SESSION_YEARS: dict[int, tuple[int, int]] = {
    131: (2015, 2016),
    132: (2017, 2018),
    133: (2019, 2020),
    134: (2021, 2022),
    135: (2023, 2024),
    136: (2025, 2026),
}
OHIO_HOUSE_SIZE: int = 99
OHIO_SENATE_SIZE: int = 33

# --- Partisan composition (approximate, for current/recent sessions) ---
# Format: {session: {"house": (R, D), "senate": (R, D), "governor_party": "R"/"D"}}
OHIO_PARTISAN_COMPOSITION: dict[int, dict] = {
    131: {"house": (65, 34), "senate": (23, 10), "governor_party": "R"},
    132: {"house": (66, 33), "senate": (24, 9), "governor_party": "R"},
    133: {"house": (61, 38), "senate": (24, 9), "governor_party": "R"},
    134: {"house": (64, 35), "senate": (25, 8), "governor_party": "R"},
    135: {"house": (67, 32), "senate": (26, 7), "governor_party": "R"},
    136: {"house": (65, 34), "senate": (24, 9), "governor_party": "R"},
}

# --- Feature Engineering ---
EARLY_INTRODUCTION_DAYS: int = 90
SESSION_DURATION_DAYS: int = 730  # ~2 years for biennial sessions

# Bill types with typically different base rates
BILL_TYPES: list[str] = ["HB", "SB", "HJR", "SJR", "HR", "SR", "HCR", "SCR"]

# LegiScan progress codes (ordinal)
PROGRESS_CODES: dict[int, str] = {
    0: "Prefiled",
    1: "Introduced",
    2: "Engrossed",  # Referred to Committee / passed committee
    3: "Enrolled",   # Passed one chamber
    4: "Passed",     # Passed both chambers
}

# LegiScan status codes
STATUS_CODES: dict[int, str] = {
    1: "Introduced",
    2: "Engrossed",
    3: "Enrolled",
    4: "Passed",
    5: "Vetoed",
    6: "Failed",
}

# Leadership positions (for feature extraction from role text)
LEADERSHIP_TITLES: list[str] = [
    "Speaker",
    "President",
    "Majority Leader",
    "Minority Leader",
    "Majority Whip",
    "Minority Whip",
    "President Pro Tempore",
    "Speaker Pro Tempore",
    "Assistant Majority Leader",
    "Assistant Minority Leader",
]

# --- Model ---
RANDOM_SEED: int = 42
TRAIN_SESSIONS: list[int] = [131, 132, 133, 134]
VALIDATION_SESSION: int = 135
CURRENT_SESSION: int = 136

# XGBoost defaults
XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

# Logistic Regression defaults
LOGISTIC_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

# Calibration
CALIBRATION_METHOD: str = "isotonic"  # "sigmoid" (Platt) or "isotonic"
CALIBRATION_CV: int = 5
