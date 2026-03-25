"""Feature engineering pipeline orchestrator.

Combines sponsor, committee, bill, and session features into a single
feature matrix suitable for model training. Handles the full pipeline:
query bills from the database, compute features, and output a DataFrame.

Usage:
    from src.features.build_features import build_feature_matrix
    df = build_feature_matrix(sessions=[131, 132, 133, 134, 135])
"""

import logging
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from src.config import OHIO_SESSION_YEARS, OHIO_SESSIONS_OF_INTEREST
from src.data.schema import Bill, Session, get_session_factory, init_database
from src.features.bill_features import compute_bill_features
from src.features.committee_features import compute_committee_features
from src.features.session_features import compute_session_features
from src.features.sponsor_features import compute_sponsor_features

logger = logging.getLogger(__name__)

# All feature columns in canonical order
FEATURE_COLUMNS: list[str] = [
    # Sponsor features
    "sponsor_majority_party",
    "sponsor_leadership",
    "sponsor_seniority",
    "sponsor_success_rate",
    "sponsor_bills_this_session",
    "sponsor_party_id",
    "sponsor_is_chair_of_committee",
    "cosponsor_count",
    "total_sponsor_count",
    "bipartisan_cosponsor_score",
    "cosponsors_on_committee",
    "cosponsor_chair_on_committee",
    "cross_chamber_cosponsors",
    # Committee features
    "committee_id",
    "committee_chamber",
    "num_committee_referrals",
    "committee_pass_through_rate",
    "committee_hearing_count",
    "committee_chair_is_sponsor",
    # Bill features
    "bill_type_encoded",
    "is_resolution",
    "is_joint_resolution",
    "originating_chamber",
    "progress",
    "status",
    "days_since_introduction",
    "days_since_last_action",
    "history_event_count",
    "early_introduction",
    "has_floor_vote",
    "roll_call_count",
    "passed_roll_call_count",
    "roll_call_success_rate",
    "amendment_count",
    "adopted_amendment_count",
    "text_length",
    "num_subjects",
    "is_appropriations",
    "has_companion",
    # Session features
    "session_pct_elapsed",
    "days_remaining_in_session",
    "is_election_year",
    "house_majority_pct",
    "senate_majority_pct",
    "governor_aligned",
    "trifecta",
    "supermajority",
    "total_session_bills",
    "base_rate_bill_type",
    "base_rate_committee",
]


def build_feature_matrix(
    sessions: list[int] | None = None,
    database_url: str | None = None,
    snapshot_date: date | None = None,
    bill_types: list[str] | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix for model training or prediction.

    Queries all bills from the specified sessions, computes all features,
    and returns a DataFrame with one row per bill.

    Args:
        sessions: Session numbers to include. Defaults to all configured.
        database_url: Override database URL. Uses default if not provided.
        snapshot_date: Compute time-dependent features as of this date.
            For training, should be the session end date.
            For prediction, should be the current date.
        bill_types: Filter to specific bill types (e.g., ["HB", "SB"]).
            Defaults to all types.

    Returns:
        DataFrame with columns: bill_id, session_id, bill_number, enacted,
        plus all feature columns from FEATURE_COLUMNS.
    """
    target_sessions = sessions or OHIO_SESSIONS_OF_INTEREST
    init_database(database_url)
    session_factory = get_session_factory(database_url)

    all_rows: list[dict[str, Any]] = []

    with session_factory() as db:
        for session_num in target_sessions:
            years = OHIO_SESSION_YEARS.get(session_num)
            if not years:
                logger.warning("No year range for session %d, skipping", session_num)
                continue

            # Find the LegiScan session_id
            session_record = db.execute(
                select(Session).where(Session.session_number == session_num)
            ).scalar_one_or_none()

            if not session_record:
                logger.warning("Session %d not found in database, skipping", session_num)
                continue

            session_id = session_record.session_id
            session_snapshot = snapshot_date or date(years[1], 12, 31)

            logger.info(
                "Building features for session %d (ID: %d), snapshot: %s",
                session_num,
                session_id,
                session_snapshot,
            )

            # Query all bills for this session
            query = select(Bill).where(Bill.session_id == session_id)
            if bill_types:
                query = query.where(Bill.bill_type.in_(bill_types))

            bills = db.execute(query).scalars().all()
            logger.info("Processing %d bills for session %d", len(bills), session_num)

            for bill in bills:
                bill_dict = _bill_to_dict(bill, years)

                try:
                    features = compute_bill_features_all(
                        db, bill_dict, session_num, session_snapshot
                    )
                    features["bill_id"] = bill.bill_id
                    features["session_id"] = session_id
                    features["session_number"] = session_num
                    features["bill_number"] = bill.bill_number
                    features["enacted"] = int(bill.enacted or False)
                    all_rows.append(features)
                except Exception as e:
                    logger.warning(
                        "Failed to compute features for bill %d (%s): %s",
                        bill.bill_id,
                        bill.bill_number,
                        e,
                    )

    if not all_rows:
        logger.warning("No bills processed — returning empty DataFrame")
        return pd.DataFrame(columns=["bill_id", "session_id", "bill_number", "enacted"] + FEATURE_COLUMNS)

    df = pd.DataFrame(all_rows)

    # Ensure all feature columns exist (fill missing with 0)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    logger.info(
        "Feature matrix: %d bills, %d features, %.1f%% enacted",
        len(df),
        len(FEATURE_COLUMNS),
        100 * df["enacted"].mean(),
    )

    return df


def compute_bill_features_all(
    db: DbSession,
    bill_row: dict,
    session_num: int,
    snapshot_date: date | None = None,
) -> dict[str, Any]:
    """Compute all features for a single bill.

    Combines sponsor, committee, bill, and session features into one dict.

    Args:
        db: Active database session.
        bill_row: Dict with bill data.
        session_num: Ohio GA session number.
        snapshot_date: Date for time-dependent features.

    Returns:
        Merged dict of all feature values.
    """
    features: dict[str, Any] = {}

    sponsor_feats = compute_sponsor_features(db, bill_row, session_num)
    features.update(sponsor_feats)

    committee_feats = compute_committee_features(db, bill_row, session_num)
    features.update(committee_feats)

    bill_feats = compute_bill_features(db, bill_row, snapshot_date)
    features.update(bill_feats)

    session_feats = compute_session_features(db, bill_row, session_num, snapshot_date)
    features.update(session_feats)

    return features


def build_single_bill_features(
    bill_id: int,
    database_url: str | None = None,
    snapshot_date: date | None = None,
) -> dict[str, Any]:
    """Compute features for a single bill (for prediction).

    Args:
        bill_id: LegiScan bill ID.
        database_url: Override database URL.
        snapshot_date: Date for time-dependent features (defaults to today).

    Returns:
        Dict of all features for this bill.

    Raises:
        ValueError: If the bill is not found in the database.
    """
    init_database(database_url)
    session_factory = get_session_factory(database_url)

    with session_factory() as db:
        bill = db.execute(
            select(Bill).where(Bill.bill_id == bill_id)
        ).scalar_one_or_none()

        if not bill:
            raise ValueError(f"Bill {bill_id} not found in database")

        session_record = db.execute(
            select(Session).where(Session.session_id == bill.session_id)
        ).scalar_one_or_none()

        if not session_record:
            raise ValueError(f"Session {bill.session_id} not found in database")

        years = OHIO_SESSION_YEARS.get(session_record.session_number, (2025, 2026))
        bill_dict = _bill_to_dict(bill, years)
        snap = snapshot_date or date.today()

        features = compute_bill_features_all(
            db, bill_dict, session_record.session_number, snap
        )
        features["bill_id"] = bill.bill_id
        features["session_id"] = bill.session_id
        features["bill_number"] = bill.bill_number
        features["enacted"] = int(bill.enacted or False)

    return features


def _bill_to_dict(bill: Bill, years: tuple[int, int]) -> dict[str, Any]:
    """Convert a Bill ORM object to a plain dict for feature functions."""
    return {
        "bill_id": bill.bill_id,
        "session_id": bill.session_id,
        "bill_number": bill.bill_number,
        "bill_type": bill.bill_type,
        "title": bill.title,
        "description": bill.description,
        "body": bill.body,
        "status": bill.status,
        "progress": bill.progress,
        "introduced_date": bill.introduced_date,
        "last_action_date": bill.last_action_date,
        "text_length": bill.text_length,
        "subject_areas": bill.subject_areas,
        "enacted": bill.enacted,
        "year_start": years[0],
    }
