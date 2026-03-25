"""Committee-related feature engineering.

Computes features about the committee(s) a bill is referred to: historical
pass-through rates, chair alignment with sponsors, hearing counts, and
party composition of the committee.
"""

import logging
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session as DbSession

from src.config import OHIO_PARTISAN_COMPOSITION
from src.data.schema import (
    Bill,
    BillHistory,
    BillSponsor,
    Committee,
    CommitteeReferral,
    Legislator,
)

logger = logging.getLogger(__name__)


def compute_committee_features(
    db: DbSession,
    bill_row: dict,
    session_num: int,
) -> dict[str, Any]:
    """Compute all committee-related features for a single bill.

    Args:
        db: Active database session.
        bill_row: Dict with bill data (must include bill_id, session_id).
        session_num: Ohio GA session number.

    Returns:
        Dict of feature name -> value.
    """
    bill_id = bill_row["bill_id"]
    session_id = bill_row["session_id"]
    features: dict[str, Any] = {}

    # Get committee referrals for this bill
    referrals = db.execute(
        select(CommitteeReferral, Committee)
        .join(Committee, CommitteeReferral.committee_id == Committee.committee_id)
        .where(CommitteeReferral.bill_id == bill_id)
    ).all()

    if not referrals:
        return _default_committee_features()

    # Use the first (primary) committee
    primary_ref, primary_comm = referrals[0]

    features["committee_id"] = primary_comm.committee_id
    features["committee_chamber"] = 1 if primary_comm.chamber == "H" else 0
    features["num_committee_referrals"] = len(referrals)
    features["committee_pass_through_rate"] = _committee_pass_through_rate(
        db, primary_comm.committee_id, session_id
    )
    features["committee_hearing_count"] = _committee_hearing_count(db, bill_id)
    features["committee_chair_is_sponsor"] = _chair_is_sponsor(
        db, bill_id, primary_comm.committee_id
    )

    return features


def _default_committee_features() -> dict[str, Any]:
    """Return default features when no committee referral exists."""
    return {
        "committee_id": 0,
        "committee_chamber": 0,
        "num_committee_referrals": 0,
        "committee_pass_through_rate": 0.0,
        "committee_hearing_count": 0,
        "committee_chair_is_sponsor": 0,
    }


def _committee_pass_through_rate(
    db: DbSession,
    committee_id: int,
    current_session_id: int,
) -> float:
    """Compute historical pass-through rate for a committee.

    Calculated as: bills that advanced past committee / total bills referred,
    using only prior sessions to prevent leakage.

    A bill is considered to have passed through committee if its progress
    is >= 2 (engrossed / passed committee) or status >= 2.
    """
    # Total bills referred to this committee in prior sessions
    total_referred = db.execute(
        select(func.count(CommitteeReferral.id))
        .join(Bill, CommitteeReferral.bill_id == Bill.bill_id)
        .where(
            CommitteeReferral.committee_id == committee_id,
            Bill.session_id < current_session_id,
        )
    ).scalar() or 0

    if total_referred == 0:
        return 0.0

    # Bills that advanced past committee (progress >= 2 means at least engrossed)
    passed_committee = db.execute(
        select(func.count(CommitteeReferral.id))
        .join(Bill, CommitteeReferral.bill_id == Bill.bill_id)
        .where(
            CommitteeReferral.committee_id == committee_id,
            Bill.session_id < current_session_id,
            Bill.progress >= 2,
        )
    ).scalar() or 0

    return passed_committee / total_referred


def _committee_hearing_count(db: DbSession, bill_id: int) -> int:
    """Count committee-related actions for this bill.

    Searches bill history for actions mentioning 'committee', 'hearing',
    'testimony', 'reported', or similar committee-activity keywords.
    """
    history_entries = db.execute(
        select(BillHistory.action)
        .where(BillHistory.bill_id == bill_id)
    ).scalars().all()

    keywords = [
        "committee", "hearing", "testimony", "reported",
        "recommends", "passage", "referred",
    ]

    count = 0
    for action in history_entries:
        action_lower = (action or "").lower()
        if any(kw in action_lower for kw in keywords):
            count += 1

    return count


def _chair_is_sponsor(
    db: DbSession,
    bill_id: int,
    committee_id: int,
) -> int:
    """Check if a committee chair/ranking member is a bill sponsor.

    Since LegiScan doesn't directly expose committee membership/chair roles,
    this checks if any sponsor has 'chair' in their role text and is
    associated with the bill's committee through other bills.
    """
    sponsors = db.execute(
        select(Legislator)
        .join(BillSponsor, Legislator.people_id == BillSponsor.people_id)
        .where(BillSponsor.bill_id == bill_id)
    ).scalars().all()

    for leg in sponsors:
        role = (leg.role or "").lower()
        if "chair" in role:
            # Check if this legislator has bills in the same committee
            has_committee_link = db.execute(
                select(func.count(CommitteeReferral.id))
                .join(BillSponsor, CommitteeReferral.bill_id == BillSponsor.bill_id)
                .where(
                    BillSponsor.people_id == leg.people_id,
                    CommitteeReferral.committee_id == committee_id,
                )
            ).scalar()
            if has_committee_link:
                return 1

    return 0
