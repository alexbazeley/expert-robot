"""Command-line interface for the legislative bill passage prediction model.

Provides commands for training, evaluation, prediction, and data management.

Usage:
    python -m src.cli predict --state OH --bill "HB 123"
    python -m src.cli predict --bill-id 1234567
    python -m src.cli train --state OH --sessions 131,132,133,134,135
    python -m src.cli evaluate --state OH --test-session 135
    python -m src.cli features --bill-id 1234567
    python -m src.cli load-data --state OH
    python -m src.cli sync --state OH --session 136
"""

import json
import logging
import sys

import click

from src.config import (
    CURRENT_SESSION,
    OHIO_SESSIONS_OF_INTEREST,
    TRAIN_SESSIONS,
    VALIDATION_SESSION,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Legislative Bill Passage Prediction Model.

    Predict the probability that a U.S. state legislature bill will be
    enacted into law, with interpretable factor attribution.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--state", default="OH", help="State abbreviation.")
@click.option("--sessions", default=None, help="Comma-separated session numbers (e.g., 131,132,133).")
@click.option("--force", is_flag=True, help="Force re-download of all data.")
def load_data(state: str, sessions: str | None, force: bool) -> None:
    """Download and load historical legislative data from LegiScan."""
    from src.data.ohio_loader import OhioDataLoader

    target_sessions = None
    if sessions:
        target_sessions = [int(s.strip()) for s in sessions.split(",")]

    click.echo(f"Loading data for {state}...")
    loader = OhioDataLoader()
    loader.load_all_sessions(sessions=target_sessions, force_reload=force)
    click.echo("Data loading complete.")


@cli.command()
@click.option("--state", default="OH", help="State abbreviation.")
@click.option("--session", default=CURRENT_SESSION, type=int, help="Session number to sync.")
def sync(state: str, session: int) -> None:
    """Incrementally sync the current session data using change_hash."""
    from src.data.legiscan_client import LegiScanClient
    from src.data.ohio_loader import OhioDataLoader
    from src.data.schema import Session as SessionModel, get_session_factory

    from sqlalchemy import select

    client = LegiScanClient()
    loader = OhioDataLoader(client=client)
    session_factory = get_session_factory()

    # Find the LegiScan session ID
    with session_factory() as db:
        session_record = db.execute(
            select(SessionModel).where(SessionModel.session_number == session)
        ).scalar_one_or_none()

    if not session_record:
        click.echo(f"Session {session} not found in database. Run load-data first.")
        sys.exit(1)

    updated = loader.sync_current_session(session_record.session_id, session)
    click.echo(f"Synced {updated} bills for session {session}.")


@cli.command()
@click.option("--state", default="OH", help="State abbreviation.")
@click.option("--sessions", default=None, help="Training session numbers (comma-separated).")
@click.option("--model-type", default="xgboost", type=click.Choice(["xgboost", "logistic"]))
@click.option("--bill-types", default=None, help="Filter bill types (comma-separated, e.g., HB,SB).")
@click.option("--tag", default="", help="Tag for model artifact filename.")
def train(state: str, sessions: str | None, model_type: str, bill_types: str | None, tag: str) -> None:
    """Train the passage prediction model on historical data."""
    from src.features.build_features import build_feature_matrix
    from src.models.passage_model import PassageModel

    train_sessions = TRAIN_SESSIONS
    if sessions:
        train_sessions = [int(s.strip()) for s in sessions.split(",")]

    types = None
    if bill_types:
        types = [t.strip() for t in bill_types.split(",")]

    click.echo(f"Building feature matrix for sessions: {train_sessions}...")
    df = build_feature_matrix(sessions=train_sessions, bill_types=types)

    if df.empty:
        click.echo("No data available. Run load-data first.")
        sys.exit(1)

    click.echo(f"Training {model_type} model on {len(df)} bills...")
    model = PassageModel(model_type=model_type)
    metadata = model.train(df)

    model_path = model.save(state=state.lower(), tag=tag)
    click.echo(f"Model saved to {model_path}")

    click.echo("\nTraining summary:")
    click.echo(f"  Bills: {metadata['n_train_total']}")
    click.echo(f"  Overall enacted rate: {metadata['overall_enacted_rate']:.1%}")
    click.echo(f"  Stage 1 positive rate: {metadata['stage1_positive_rate']:.1%}")
    click.echo(f"  Stage 2 positive rate: {metadata['stage2_positive_rate']:.1%}")


@cli.command()
@click.option("--state", default="OH", help="State abbreviation.")
@click.option("--test-session", default=VALIDATION_SESSION, type=int, help="Session to test on.")
@click.option("--model-type", default="xgboost", type=click.Choice(["xgboost", "logistic"]))
@click.option("--tag", default="", help="Model tag to evaluate.")
def evaluate(state: str, test_session: int, model_type: str, tag: str) -> None:
    """Evaluate model on a held-out session (temporal validation)."""
    from src.features.build_features import build_feature_matrix
    from src.models.evaluate import evaluate_model
    from src.models.passage_model import PassageModel

    click.echo(f"Building features for test session {test_session}...")
    test_df = build_feature_matrix(sessions=[test_session])

    if test_df.empty:
        click.echo(f"No data for session {test_session}. Run load-data first.")
        sys.exit(1)

    click.echo(f"Loading {model_type} model...")
    model = PassageModel.load(state=state.lower(), model_type=model_type, tag=tag)

    click.echo(f"Evaluating on {len(test_df)} bills...")
    metrics = evaluate_model(model, test_df)

    click.echo("\nEvaluation results:")
    click.echo(f"  Test bills: {metrics['n_test']}")
    click.echo(f"  Enacted: {metrics['n_positive']} ({metrics['base_rate']:.1%})")
    if metrics.get("auc_roc") is not None:
        click.echo(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        click.echo(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    click.echo(f"  Brier:   {metrics['brier_score']:.4f}")
    click.echo(f"  Log Loss: {metrics['log_loss']:.4f}")
    click.echo(f"  ECE:     {metrics['calibration']['ece']:.4f}")


@cli.command()
@click.option("--state", default="OH", help="State abbreviation.")
@click.option("--bill", default=None, help="Bill number (e.g., 'HB 123').")
@click.option("--bill-id", default=None, type=int, help="LegiScan bill ID.")
@click.option("--model-type", default="xgboost", type=click.Choice(["xgboost", "logistic"]))
def predict(state: str, bill: str | None, bill_id: int | None, model_type: str) -> None:
    """Predict passage probability for a specific bill."""
    from src.predict import predict_bill

    if bill is None and bill_id is None:
        click.echo("Must provide either --bill or --bill-id")
        sys.exit(1)

    try:
        result = predict_bill(
            bill_id=bill_id,
            state=state,
            bill_number=bill,
            model_type=model_type,
        )
    except FileNotFoundError:
        click.echo("No trained model found. Run 'train' first.")
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    click.echo(result["narrative"])


@cli.command()
@click.option("--bill-id", required=True, type=int, help="LegiScan bill ID.")
@click.option("--json-output", is_flag=True, help="Output features as JSON.")
def features(bill_id: int, json_output: bool) -> None:
    """Inspect computed features for a specific bill."""
    from src.features.build_features import FEATURE_COLUMNS, build_single_bill_features

    try:
        feature_dict = build_single_bill_features(bill_id)
    except ValueError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    if json_output:
        output = {k: v for k, v in feature_dict.items() if k in FEATURE_COLUMNS}
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        click.echo(f"\nFeatures for bill {bill_id} ({feature_dict.get('bill_number', '')}):\n")
        for col in FEATURE_COLUMNS:
            val = feature_dict.get(col, "N/A")
            if isinstance(val, float):
                click.echo(f"  {col:40s} {val:.4f}")
            else:
                click.echo(f"  {col:40s} {val}")


if __name__ == "__main__":
    cli()
