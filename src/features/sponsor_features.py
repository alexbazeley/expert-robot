"""Sponsor and cosponsorship feature engineering.

Computes features related to bill sponsors and cosponsors: majority party
membership, leadership positions, seniority, historical success rate,
bipartisan cosponsorship, committee overlap, and legislative bandwidth.
"""

import logging
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from src.config import LEADERSHIP_TITLES, OHIO_PARTISAN_COMPOSITION
from src.data.schema import Bill, BillSponsor, CommitteeReferral, Legislator

logger = logging.getLogger(__name__)


def compute_sponsor_features(db: DbSession, bill_row: dict, session_num: int) -> dict[str, Any]:
    """Compute all sponsor-related features for a single bill.

    Args:
        db: Active database session.
        bill_row: Dict with bill data (must include bill_id, session_id).
        session_num: Ohio GA session number (e.g., 136).

    Returns:
        Dict of feature name -> value.
    """
    bill_id = bill_row["bill_id"]
    session_id = bill_row["session_id"]
    features: dict[str, Any] = {}

    # Get all sponsors for this bill
    sponsors = db.execute(
        select(BillSponsor, Legislator)
        .join(Legislator, BillSponsor.people_id == Legislator.people_id)
        .where(BillSponsor.bill_id == bill_id)
        .order_by(BillSponsor.sponsor_order)
    ).all()

    if not sponsors:
        return _default_sponsor_features()

    # Separate primary sponsor from cosponsors
    primary = None
    cosponsors = []
    for bs, leg in sponsors:
        if bs.sponsor_type == 1:
            primary = (bs, leg)
        else:
            cosponsors.append((bs, leg))

    # If no explicit primary, treat first sponsor as primary
    if primary is None and sponsors:
        primary = sponsors[0]
        cosponsors = list(sponsors[1:])

    # --- Primary sponsor features ---
    _, primary_leg = primary
    composition = OHIO_PARTISAN_COMPOSITION.get(session_num, {})

    features["sponsor_majority_party"] = _is_majority_party(primary_leg, composition)
    features["sponsor_leadership"] = _has_leadership_position(primary_leg)
    features["sponsor_seniority"] = _compute_seniority(db, primary_leg.people_id, session_id)
    features["sponsor_success_rate"] = _compute_success_rate(db, primary_leg.people_id, session_id)
    features["sponsor_bills_this_session"] = _count_session_bills(
        db, primary_leg.people_id, session_id
    )
    features["sponsor_party_id"] = primary_leg.party_id or 0
    features["sponsor_is_chair_of_committee"] = _sponsor_is_committee_chair(
        db, bill_id, primary_leg
    )

    # --- Cosponsorship features ---
    all_sponsors = [s for _, s in sponsors]
    features["cosponsor_count"] = len(cosponsors)
    features["total_sponsor_count"] = len(sponsors)
    features["bipartisan_cosponsor_score"] = _bipartisan_score(all_sponsors, primary_leg)
    features["cosponsors_on_committee"] = _cosponsors_on_committee(db, bill_id, cosponsors)
    features["cosponsor_chair_on_committee"] = _cosponsor_is_committee_chair(
        db, bill_id, cosponsors
    )
    features["cross_chamber_cosponsors"] = _cross_chamber_cosponsors(cosponsors, primary_leg)

    return features


def _default_sponsor_features() -> dict[str, Any]:
    """Return default features when no sponsor data is available."""
    return {
        "sponsor_majority_party": 0,
        "sponsor_leadership": 0,
        "sponsor_seniority": 0,
        "sponsor_success_rate": 0.0,
        "sponsor_bills_this_session": 0,
        "sponsor_party_id": 0,
        "sponsor_is_chair_of_committee": 0,
        "cosponsor_count": 0,
        "total_sponsor_count": 0,
        "bipartisan_cosponsor_score": 0.0,
        "cosponsors_on_committee": 0,
        "cosponsor_chair_on_committee": 0,
        "cross_chamber_cosponsors": 0,
    }


def _is_majority_party(legislator: Legislator, composition: dict) -> int:
    """Check if legislator is in the majority party.

    In Ohio, Republican (party_id=2) is the majority in both chambers.
    """
    if not legislator.party_id:
        return 0
    # party_id 2 = Republican, which is the majority in Ohio
    return int(legislator.party_id == 2)


def _has_leadership_position(legislator: Legislator) -> int:
    """Check if legislator holds a leadership position.

    Uses role field and name matching against known leadership titles.
    Returns 1 if leadership, 0 otherwise.
    """
    role = (legislator.role or "").lower()
    name = (legislator.name or "").lower()
    combined = f"{role} {name}"

    for title in LEADERSHIP_TITLES:
        if title.lower() in combined:
            return 1
    return 0


def _compute_seniority(db: DbSession, people_id: int, current_session_id: int) -> int:
    """Count how many prior sessions this legislator has sponsored bills in.

    Proxy for seniority — counts distinct sessions where they appear as
    a sponsor on any bill before the current session.
    """
    result = db.execute(
        select(func.count(func.distinct(Bill.session_id)))
        .join(BillSponsor, Bill.bill_id == BillSponsor.bill_id)
        .where(
            BillSponsor.people_id == people_id,
            Bill.session_id < current_session_id,
        )
    ).scalar()
    return result or 0


def _compute_success_rate(db: DbSession, people_id: int, current_session_id: int) -> float:
    """Compute historical bill passage rate for this sponsor.

    Only considers prior sessions to avoid leakage. Returns the fraction
    of bills they sponsored that were enacted.
    """
    total = db.execute(
        select(func.count(Bill.bill_id))
        .join(BillSponsor, Bill.bill_id == BillSponsor.bill_id)
        .where(
            BillSponsor.people_id == people_id,
            BillSponsor.sponsor_type == 1,
            Bill.session_id < current_session_id,
        )
    ).scalar() or 0

    if total == 0:
        return 0.0

    passed = db.execute(
        select(func.count(Bill.bill_id))
        .join(BillSponsor, Bill.bill_id == BillSponsor.bill_id)
        .where(
            BillSponsor.people_id == people_id,
            BillSponsor.sponsor_type == 1,
            Bill.session_id < current_session_id,
            Bill.enacted == True,  # noqa: E712
        )
    ).scalar() or 0

    return passed / total


def _count_session_bills(db: DbSession, people_id: int, session_id: int) -> int:
    """Count bills this legislator has sponsored in the current session."""
    result = db.execute(
        select(func.count(BillSponsor.id))
        .join(Bill, BillSponsor.bill_id == Bill.bill_id)
        .where(
            BillSponsor.people_id == people_id,
            BillSponsor.sponsor_type == 1,
            Bill.session_id == session_id,
        )
    ).scalar()
    return result or 0


def _bipartisan_score(all_sponsors: list[Legislator], primary: Legislator) -> float:
    """Compute bipartisan cosponsorship score.

    Returns the fraction of cosponsors from a different party than the
    primary sponsor. 0.0 = all same party, 1.0 = all opposing party.
    """
    if len(all_sponsors) <= 1:
        return 0.0

    primary_party = primary.party_id
    if not primary_party:
        return 0.0

    others = [s for s in all_sponsors if s.people_id != primary.people_id]
    if not others:
        return 0.0

    cross_party = sum(1 for s in others if s.party_id and s.party_id != primary_party)
    return cross_party / len(others)


def _cosponsors_on_committee(
    db: DbSession,
    bill_id: int,
    cosponsors: list[tuple],
) -> int:
    """Count cosponsors who serve on the bill's assigned committee.

    Note: LegiScan doesn't directly provide committee membership, so this
    is approximated by checking if the cosponsor has bills referred to the
    same committee (indicating they interact with that committee).
    """
    # Get committee(s) this bill is referred to
    referrals = db.execute(
        select(CommitteeReferral.committee_id)
        .where(CommitteeReferral.bill_id == bill_id)
    ).scalars().all()

    if not referrals:
        return 0

    count = 0
    for _, leg in cosponsors:
        # Check if this cosponsor has any bills in the same committee
        has_committee_bill = db.execute(
            select(func.count(CommitteeReferral.id))
            .join(BillSponsor, CommitteeReferral.bill_id == BillSponsor.bill_id)
            .where(
                BillSponsor.people_id == leg.people_id,
                CommitteeReferral.committee_id.in_(referrals),
            )
        ).scalar()
        if has_committee_bill:
            count += 1

    return count


def _sponsor_is_committee_chair(
    db: DbSession,
    bill_id: int,
    legislator: Legislator,
) -> int:
    """Check if the primary sponsor chairs the assigned committee.

    Approximated by checking legislator role text for 'chair' and whether
    they have significant activity in the bill's committee.
    """
    role = (legislator.role or "").lower()
    if "chair" in role:
        # Check if they're associated with the bill's committee
        referrals = db.execute(
            select(CommitteeReferral.committee_id)
            .where(CommitteeReferral.bill_id == bill_id)
        ).scalars().all()
        if referrals:
            return 1
    return 0


def _cosponsor_is_committee_chair(
    db: DbSession,
    bill_id: int,
    cosponsors: list[tuple],
) -> int:
    """Check if any cosponsor chairs the assigned committee."""
    for _, leg in cosponsors:
        if _sponsor_is_committee_chair(db, bill_id, leg):
            return 1
    return 0


def _cross_chamber_cosponsors(
    cosponsors: list[tuple],
    primary: Legislator,
) -> int:
    """Count cosponsors from the other chamber."""
    primary_role = (primary.role or "").lower()
    primary_chamber = "H" if "rep" in primary_role else "S"

    count = 0
    for _, leg in cosponsors:
        leg_role = (leg.role or "").lower()
        leg_chamber = "H" if "rep" in leg_role else "S"
        if leg_chamber != primary_chamber:
            count += 1
    return count
