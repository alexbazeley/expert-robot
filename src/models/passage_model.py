"""Bill passage prediction model.

Implements a two-stage prediction architecture:
    Stage 1: P(bill exits committee)
    Stage 2: P(bill is enacted | exited committee)
    Combined: P(enacted) = P(exits committee) * P(enacted | exits committee)

Supports XGBoost (primary) and Logistic Regression (interpretable baseline)
with probability calibration via Platt scaling or isotonic regression.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_CV,
    CALIBRATION_METHOD,
    LOGISTIC_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    XGBOOST_PARAMS,
)
from src.features.build_features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

ModelType = Literal["xgboost", "logistic"]


class PassageModel:
    """Two-stage bill passage prediction model with calibration.

    Stage 1 predicts whether a bill exits committee (progress >= 2).
    Stage 2 predicts whether a bill is enacted, given it exited committee.
    The combined probability is the product of the two stages.

    Args:
        model_type: "xgboost" or "logistic".
        calibrate: Whether to apply probability calibration.
        calibration_method: "sigmoid" (Platt) or "isotonic".
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        calibrate: bool = True,
        calibration_method: str | None = None,
    ) -> None:
        self.model_type = model_type
        self.calibrate = calibrate
        self.calibration_method = calibration_method or CALIBRATION_METHOD
        self.feature_columns = FEATURE_COLUMNS.copy()

        self.stage1_model: Any = None  # P(exits committee)
        self.stage2_model: Any = None  # P(enacted | exits committee)
        self.is_fitted: bool = False
        self.metadata: dict[str, Any] = {}

    def _create_base_model(self, pos_weight: float = 1.0) -> Any:
        """Create a fresh base model instance.

        Args:
            pos_weight: Scale factor for positive class weight (for
                class imbalance handling in XGBoost).

        Returns:
            Unfitted model instance.
        """
        if self.model_type == "xgboost":
            params = XGBOOST_PARAMS.copy()
            params["scale_pos_weight"] = pos_weight
            return XGBClassifier(**params)
        else:
            params = LOGISTIC_PARAMS.copy()
            return LogisticRegression(class_weight="balanced", **params)

    def _create_calibrated_model(self, base_model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Wrap a fitted model with probability calibration.

        Uses cross-validated calibration to avoid overfitting the
        calibration curve to the training data.
        """
        if not self.calibrate:
            return base_model

        n_positive = int(y.sum())
        cv_folds = min(CALIBRATION_CV, n_positive) if n_positive > 1 else 2

        if cv_folds < 2:
            logger.warning("Too few positive samples for calibration, skipping")
            return base_model

        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method=self.calibration_method,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED),
        )
        calibrated.fit(X, y)
        return calibrated

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train both stages of the model.

        Args:
            df: Feature matrix from build_feature_matrix(). Must contain
                columns: 'enacted', 'progress', plus all FEATURE_COLUMNS.

        Returns:
            Training metadata dict with class balance info and metrics.
        """
        logger.info("Training %s model on %d bills", self.model_type, len(df))

        # Validate input
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[self.feature_columns].values.astype(np.float32)
        feature_names = self.feature_columns

        # --- Stage 1: Exits committee (progress >= 2) ---
        y_committee = (df["progress"] >= 2).astype(int).values
        n_pos_committee = int(y_committee.sum())
        n_neg_committee = len(y_committee) - n_pos_committee

        logger.info(
            "Stage 1 — exits committee: %d positive / %d negative (%.1f%%)",
            n_pos_committee,
            n_neg_committee,
            100 * n_pos_committee / len(y_committee) if len(y_committee) > 0 else 0,
        )

        pos_weight_1 = n_neg_committee / max(n_pos_committee, 1)
        base_model_1 = self._create_base_model(pos_weight_1)
        base_model_1.fit(X, y_committee)
        self.stage1_model = self._create_calibrated_model(base_model_1, X, y_committee)

        # --- Stage 2: Enacted given exited committee ---
        committee_mask = y_committee == 1
        if committee_mask.sum() < 10:
            logger.warning(
                "Only %d bills exited committee — Stage 2 may be unreliable",
                committee_mask.sum(),
            )

        X_stage2 = X[committee_mask]
        y_enacted = df.loc[committee_mask.astype(bool), "enacted"].astype(int).values

        n_pos_enacted = int(y_enacted.sum())
        n_neg_enacted = len(y_enacted) - n_pos_enacted

        logger.info(
            "Stage 2 — enacted | committee: %d positive / %d negative (%.1f%%)",
            n_pos_enacted,
            n_neg_enacted,
            100 * n_pos_enacted / len(y_enacted) if len(y_enacted) > 0 else 0,
        )

        pos_weight_2 = n_neg_enacted / max(n_pos_enacted, 1)
        base_model_2 = self._create_base_model(pos_weight_2)
        base_model_2.fit(X_stage2, y_enacted)
        self.stage2_model = self._create_calibrated_model(base_model_2, X_stage2, y_enacted)

        self.is_fitted = True
        self.metadata = {
            "model_type": self.model_type,
            "calibration": self.calibration_method if self.calibrate else "none",
            "n_train_total": len(df),
            "n_train_stage2": int(committee_mask.sum()),
            "stage1_positive_rate": n_pos_committee / len(y_committee) if len(y_committee) else 0,
            "stage2_positive_rate": n_pos_enacted / len(y_enacted) if len(y_enacted) else 0,
            "overall_enacted_rate": int(df["enacted"].sum()) / len(df) if len(df) else 0,
            "n_features": len(self.feature_columns),
            "trained_at": datetime.utcnow().isoformat(),
            "random_seed": RANDOM_SEED,
        }

        logger.info("Training complete. Metadata: %s", self.metadata)
        return self.metadata

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict passage probability for bills.

        Returns the combined two-stage probability:
        P(enacted) = P(exits committee) * P(enacted | exits committee)

        Args:
            df: Feature matrix with FEATURE_COLUMNS.

        Returns:
            1D array of passage probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X = df[self.feature_columns].values.astype(np.float32)

        p_committee = self.stage1_model.predict_proba(X)[:, 1]
        p_enacted_given_committee = self.stage2_model.predict_proba(X)[:, 1]

        # Combined probability
        p_enacted = p_committee * p_enacted_given_committee

        # Clip to valid probability range
        return np.clip(p_enacted, 0.0, 1.0)

    def predict_stages(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict both stage probabilities separately (for analysis).

        Returns:
            Dict with keys: "p_committee", "p_enacted_given_committee", "p_enacted".
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X = df[self.feature_columns].values.astype(np.float32)

        p_committee = self.stage1_model.predict_proba(X)[:, 1]
        p_enacted_given_committee = self.stage2_model.predict_proba(X)[:, 1]
        p_enacted = np.clip(p_committee * p_enacted_given_committee, 0.0, 1.0)

        return {
            "p_committee": p_committee,
            "p_enacted_given_committee": p_enacted_given_committee,
            "p_enacted": p_enacted,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from the stage 1 model.

        For XGBoost, uses gain-based importance. For logistic regression,
        uses absolute coefficient values.

        Returns:
            DataFrame with columns: feature, importance, sorted descending.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # Get the base estimator from calibrated wrapper if needed
        model = self.stage1_model
        if hasattr(model, "estimator"):
            model = model.estimator
        elif hasattr(model, "calibrated_classifiers_"):
            model = model.calibrated_classifiers_[0].estimator

        if self.model_type == "xgboost":
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])

        df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, state: str = "ohio", tag: str = "") -> Path:
        """Save the trained model and metadata to disk.

        Args:
            state: State identifier for the output directory.
            tag: Optional tag to append to filename.

        Returns:
            Path to the saved model file.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        model_dir = MODELS_DIR / state
        model_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        model_path = model_dir / f"passage_model_{self.model_type}{suffix}.joblib"
        meta_path = model_dir / f"passage_model_{self.model_type}{suffix}_meta.json"

        joblib.dump({
            "stage1_model": self.stage1_model,
            "stage2_model": self.stage2_model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
            "calibrate": self.calibrate,
            "calibration_method": self.calibration_method,
        }, model_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Model saved to %s", model_path)
        return model_path

    @classmethod
    def load(cls, state: str = "ohio", model_type: ModelType = "xgboost", tag: str = "") -> "PassageModel":
        """Load a trained model from disk.

        Args:
            state: State identifier.
            model_type: Model type to load.
            tag: Optional tag suffix.

        Returns:
            Loaded PassageModel instance.
        """
        model_dir = MODELS_DIR / state
        suffix = f"_{tag}" if tag else ""
        model_path = model_dir / f"passage_model_{model_type}{suffix}.joblib"
        meta_path = model_dir / f"passage_model_{model_type}{suffix}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        data = joblib.load(model_path)

        instance = cls(
            model_type=data["model_type"],
            calibrate=data["calibrate"],
            calibration_method=data["calibration_method"],
        )
        instance.stage1_model = data["stage1_model"]
        instance.stage2_model = data["stage2_model"]
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = True

        if meta_path.exists():
            with open(meta_path) as f:
                instance.metadata = json.load(f)

        logger.info("Model loaded from %s", model_path)
        return instance
