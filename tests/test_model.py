"""Smoke tests for the passage prediction model.

Tests model training, prediction, save/load, and evaluation with
synthetic data to verify the pipeline works end-to-end without
requiring real legislative data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS
from src.models.evaluate import compute_calibration, evaluate_model
from src.models.passage_model import PassageModel


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Generate synthetic training data that mimics real bill features.

    Creates a dataset with realistic class imbalance (~10% positive)
    and features that have some signal for the target.
    """
    rng = np.random.RandomState(42)
    n = 500

    data: dict[str, np.ndarray] = {}

    # Generate features with some correlation to outcome
    data["sponsor_majority_party"] = rng.binomial(1, 0.6, n)
    data["sponsor_leadership"] = rng.binomial(1, 0.1, n)
    data["sponsor_seniority"] = rng.poisson(2, n)
    data["sponsor_success_rate"] = rng.beta(2, 8, n)
    data["sponsor_bills_this_session"] = rng.poisson(5, n)
    data["sponsor_party_id"] = rng.choice([1, 2], n, p=[0.4, 0.6])
    data["sponsor_is_chair_of_committee"] = rng.binomial(1, 0.05, n)
    data["cosponsor_count"] = rng.poisson(3, n)
    data["total_sponsor_count"] = data["cosponsor_count"] + 1
    data["bipartisan_cosponsor_score"] = rng.beta(1, 5, n)
    data["cosponsors_on_committee"] = rng.poisson(1, n)
    data["cosponsor_chair_on_committee"] = rng.binomial(1, 0.03, n)
    data["cross_chamber_cosponsors"] = rng.poisson(0.5, n)
    data["committee_id"] = rng.randint(1, 30, n)
    data["committee_chamber"] = rng.binomial(1, 0.5, n)
    data["num_committee_referrals"] = rng.poisson(1, n) + 1
    data["committee_pass_through_rate"] = rng.beta(3, 7, n)
    data["committee_hearing_count"] = rng.poisson(2, n)
    data["committee_chair_is_sponsor"] = rng.binomial(1, 0.05, n)
    data["bill_type_encoded"] = rng.choice([1, 2, 3, 4, 5], n, p=[0.4, 0.3, 0.1, 0.1, 0.1])
    data["is_resolution"] = (data["bill_type_encoded"] >= 3).astype(int)
    data["is_joint_resolution"] = (data["bill_type_encoded"].astype(int) == 3).astype(int)
    data["originating_chamber"] = rng.binomial(1, 0.6, n)
    data["progress"] = rng.choice([0, 1, 2, 3, 4], n, p=[0.05, 0.6, 0.2, 0.1, 0.05])
    data["status"] = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.15])
    data["days_since_introduction"] = rng.exponential(100, n).astype(int)
    data["days_since_last_action"] = rng.exponential(30, n).astype(int)
    data["history_event_count"] = rng.poisson(4, n)
    data["early_introduction"] = rng.binomial(1, 0.4, n)
    data["has_floor_vote"] = rng.binomial(1, 0.15, n)
    data["roll_call_count"] = data["has_floor_vote"] * rng.poisson(1, n)
    data["passed_roll_call_count"] = (data["roll_call_count"] * rng.beta(7, 3, n)).astype(int)
    data["roll_call_success_rate"] = np.where(
        data["roll_call_count"] > 0,
        data["passed_roll_call_count"] / data["roll_call_count"],
        0.0,
    )
    data["amendment_count"] = rng.poisson(1, n)
    data["adopted_amendment_count"] = (data["amendment_count"] * rng.beta(5, 5, n)).astype(int)
    data["text_length"] = rng.exponential(10000, n).astype(int)
    data["num_subjects"] = rng.poisson(2, n)
    data["is_appropriations"] = rng.binomial(1, 0.1, n)
    data["has_companion"] = rng.binomial(1, 0.1, n)
    data["session_pct_elapsed"] = rng.uniform(0.1, 0.9, n)
    data["days_remaining_in_session"] = ((1 - data["session_pct_elapsed"]) * 730).astype(int)
    data["is_election_year"] = rng.binomial(1, 0.5, n)
    data["house_majority_pct"] = rng.uniform(0.55, 0.7, n)
    data["senate_majority_pct"] = rng.uniform(0.6, 0.75, n)
    data["governor_aligned"] = rng.binomial(1, 0.7, n)
    data["trifecta"] = rng.binomial(1, 0.6, n)
    data["supermajority"] = rng.binomial(1, 0.4, n)
    data["total_session_bills"] = rng.randint(500, 2000, n)
    data["base_rate_bill_type"] = rng.beta(1, 10, n)
    data["base_rate_committee"] = rng.beta(1, 8, n)

    # Generate target with realistic signal
    logit = (
        0.5 * data["progress"]
        + 0.3 * data["sponsor_majority_party"]
        + 0.2 * data["sponsor_leadership"]
        + 0.1 * data["cosponsor_count"]
        - 0.01 * data["days_since_last_action"]
        + 0.2 * data["has_floor_vote"]
        - 2.5  # base rate offset
    )
    prob = 1 / (1 + np.exp(-logit))
    data["enacted"] = rng.binomial(1, prob)

    df = pd.DataFrame(data)

    # Verify all feature columns are present
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"

    return df


class TestPassageModel:
    def test_train_xgboost(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=False)
        metadata = model.train(synthetic_data)

        assert model.is_fitted
        assert metadata["n_train_total"] == len(synthetic_data)
        assert metadata["model_type"] == "xgboost"
        assert 0 < metadata["overall_enacted_rate"] < 1

    def test_train_logistic(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="logistic", calibrate=False)
        metadata = model.train(synthetic_data)

        assert model.is_fitted
        assert metadata["model_type"] == "logistic"

    def test_predict_returns_probabilities(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=False)
        model.train(synthetic_data)

        probs = model.predict_proba(synthetic_data)
        assert len(probs) == len(synthetic_data)
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_stages(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=False)
        model.train(synthetic_data)

        stages = model.predict_stages(synthetic_data)
        assert "p_committee" in stages
        assert "p_enacted_given_committee" in stages
        assert "p_enacted" in stages
        assert len(stages["p_enacted"]) == len(synthetic_data)

    def test_feature_importance(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=False)
        model.train(synthetic_data)

        importance = model.get_feature_importance()
        assert len(importance) == len(FEATURE_COLUMNS)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert importance["importance"].sum() > 0

    def test_save_and_load(self, synthetic_data: pd.DataFrame, tmp_path: Path) -> None:
        from src.config import MODELS_DIR

        model = PassageModel(model_type="xgboost", calibrate=False)
        model.train(synthetic_data)

        # Save
        model_path = model.save(state="test", tag="smoke")
        assert model_path.exists()

        # Load
        loaded = PassageModel.load(state="test", model_type="xgboost", tag="smoke")
        assert loaded.is_fitted

        # Predictions should match
        orig_probs = model.predict_proba(synthetic_data)
        loaded_probs = loaded.predict_proba(synthetic_data)
        np.testing.assert_array_almost_equal(orig_probs, loaded_probs)

    def test_unfitted_model_raises(self) -> None:
        model = PassageModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict_proba(pd.DataFrame())

    def test_missing_features_raises(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=False)
        bad_df = synthetic_data.drop(columns=["progress"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            model.train(bad_df)

    def test_calibrated_model(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=True, calibration_method="sigmoid")
        model.train(synthetic_data)

        probs = model.predict_proba(synthetic_data)
        assert all(0 <= p <= 1 for p in probs)


class TestEvaluation:
    def test_evaluate_returns_metrics(self, synthetic_data: pd.DataFrame) -> None:
        model = PassageModel(model_type="xgboost", calibrate=False)
        model.train(synthetic_data)

        metrics = evaluate_model(model, synthetic_data)
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "calibration" in metrics

    def test_calibration_computation(self) -> None:
        y_true = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.1, 0.8, 0.3, 0.7, 0.2, 0.1, 0.6, 0.15])

        cal = compute_calibration(y_true, y_prob, n_bins=5)
        assert "bins" in cal
        assert "ece" in cal
        assert len(cal["bins"]) == 5
        assert 0 <= cal["ece"] <= 1

    def test_evaluation_with_all_negative(self) -> None:
        """Model should handle edge case of all-negative test set."""
        rng = np.random.RandomState(42)
        n = 100
        data = {col: rng.random(n) for col in FEATURE_COLUMNS}
        data["enacted"] = np.zeros(n, dtype=int)
        data["progress"] = rng.randint(0, 3, n)
        df = pd.DataFrame(data)

        model = PassageModel(model_type="xgboost", calibrate=False)
        # Need some positives for training
        train_data = df.copy()
        train_data.loc[:9, "enacted"] = 1
        train_data.loc[:9, "progress"] = 4
        model.train(train_data)

        metrics = evaluate_model(model, df)
        assert metrics["n_positive"] == 0
        assert metrics["auc_roc"] is None
