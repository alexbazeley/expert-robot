"""Session-level contextual feature engineering.

Computes features about the legislative session context: time remaining,
election year indicator, partisan composition, governor alignment,
legislative congestion, and historical base rates by bill type,
committee, and subject.
"""

import logging
from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from src.config import (
    OHIO_PARTISAN_COMPOSITION,
    OHIO_SESSION_YEARS,
    SESSION_DURATION_DAYS,
)
from src.data.schema import Bill, CommitteeReferral

logger = logging.getLogger(__name__)


def compute_session_features(
    db: DbSession,
    bill_row: dict,
    session_num: int,
    snapshot_date: date | None = None,
) -> dict[str, Any]:
    """Compute all session-context features for a single bill.

    Args:
        db: Active database session.
        bill_row: Dict with bill data.
        session_num: Ohio GA session number.
        snapshot_date: Date to compute time-dependent features as of.

    Returns:
        Dict of feature name -> value.
    """
    session_id = bill_row["session_id"]
    features: dict[str, Any] = {}

    ref_date = snapshot_date or date.today()
    years = OHIO_SESSION_YEARS.get(session_num, (2025, 2026))
    composition = OHIO_PARTISAN_COMPOSITION.get(session_num, {})

    # --- Time in session ---
    session_start = date(years[0], 1, 1)
    session_end = date(years[1], 12, 31)

    total_days = (session_end - session_start).days
    elapsed_days = max(0, (ref_date - session_start).days)
    features["session_pct_elapsed"] = min(1.0, elapsed_days / total_days) if total_days > 0 else 0.0
    features["days_remaining_in_session"] = max(0, (session_end - ref_date).days)

    # --- Election year ---
    # Ohio: even years are election years for state legislators
    current_year = ref_date.year if ref_date else years[1]
    features["is_election_year"] = int(current_year % 2 == 0)

    # --- Partisan composition ---
    house_comp = composition.get("house", (0, 0))
    senate_comp = composition.get("senate", (0, 0))
    gov_party = composition.get("governor_party", "")

    house_total = sum(house_comp)
    senate_total = sum(senate_comp)

    features["house_majority_pct"] = house_comp[0] / house_total if house_total > 0 else 0.0
    features["senate_majority_pct"] = senate_comp[0] / senate_total if senate_total > 0 else 0.0
    features["governor_aligned"] = int(gov_party == "R")  # majority is R in Ohio
    features["trifecta"] = int(
        features["house_majority_pct"] > 0.5
        and features["senate_majority_pct"] > 0.5
        and features["governor_aligned"] == 1
    )
    features["supermajority"] = int(
        (house_comp[0] / house_total if house_total else 0) >= 2 / 3
        and (senate_comp[0] / senate_total if senate_total else 0) >= 2 / 3
    )

    # --- Legislative congestion ---
    total_bills = db.execute(
        select(func.count(Bill.bill_id))
        .where(Bill.session_id == session_id)
    ).scalar() or 0
    features["total_session_bills"] = total_bills

    # --- Historical base rates (from prior sessions) ---
    bill_type = bill_row.get("bill_type", "")
    features["base_rate_bill_type"] = _base_rate_by_type(db, bill_type, session_id)
    features["base_rate_committee"] = _base_rate_by_committee(db, bill_row["bill_id"], session_id)

    return features


def _base_rate_by_type(db: DbSession, bill_type: str, current_session_id: int) -> float:
    """Compute historical passage rate for this bill type across prior sessions."""
    if not bill_type:
        return 0.0

    total = db.execute(
        select(func.count(Bill.bill_id))
        .where(
            Bill.bill_type == bill_type,
            Bill.session_id < current_session_id,
        )
    ).scalar() or 0

    if total == 0:
        return 0.0

    passed = db.execute(
        select(func.count(Bill.bill_id))
        .where(
            Bill.bill_type == bill_type,
            Bill.session_id < current_session_id,
            Bill.enacted == True,  # noqa: E712
        )
    ).scalar() or 0

    return passed / total


def _base_rate_by_committee(db: DbSession, bill_id: int, current_session_id: int) -> float:
    """Compute historical passage rate for this bill's committee across prior sessions."""
    # Get the committee this bill is referred to
    referral = db.execute(
        select(CommitteeReferral.committee_id)
        .where(CommitteeReferral.bill_id == bill_id)
    ).scalar()

    if not referral:
        return 0.0

    total = db.execute(
        select(func.count(CommitteeReferral.id))
        .join(Bill, CommitteeReferral.bill_id == Bill.bill_id)
        .where(
            CommitteeReferral.committee_id == referral,
            Bill.session_id < current_session_id,
        )
    ).scalar() or 0

    if total == 0:
        return 0.0

    passed = db.execute(
        select(func.count(CommitteeReferral.id))
        .join(Bill, CommitteeReferral.bill_id == Bill.bill_id)
        .where(
            CommitteeReferral.committee_id == referral,
            Bill.session_id < current_session_id,
            Bill.enacted == True,  # noqa: E712
        )
    ).scalar() or 0

    return passed / total
