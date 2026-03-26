"""Bill-level feature engineering.

Computes features intrinsic to the bill itself: lifecycle/momentum indicators,
content characteristics, type, amendments, roll call results, and companion
bill detection.
"""

import json
import logging
from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from src.config import EARLY_INTRODUCTION_DAYS, SESSION_DURATION_DAYS
from src.data.schema import Amendment, Bill, BillHistory, RollCall

logger = logging.getLogger(__name__)

# Bill type encoding: maps type prefix to an ordinal for the model
BILL_TYPE_MAP: dict[str, int] = {
    "HB": 1,
    "SB": 2,
    "HJR": 3,
    "SJR": 4,
    "HR": 5,
    "SR": 6,
    "HCR": 7,
    "SCR": 8,
}


def compute_bill_features(
    db: DbSession,
    bill_row: dict,
    snapshot_date: date | None = None,
) -> dict[str, Any]:
    """Compute all bill-level features.

    Args:
        db: Active database session.
        bill_row: Dict with bill data from the bills table.
        snapshot_date: Date to compute time-dependent features as of.
            Defaults to bill's last_action_date or today.

    Returns:
        Dict of feature name -> value.
    """
    bill_id = bill_row["bill_id"]
    features: dict[str, Any] = {}

    # Use snapshot_date for time-dependent features
    ref_date = snapshot_date
    if ref_date is None:
        ref_date = bill_row.get("last_action_date") or date.today()
    if isinstance(ref_date, str):
        from datetime import datetime
        ref_date = datetime.strptime(ref_date, "%Y-%m-%d").date()

    introduced_date = bill_row.get("introduced_date")
    if isinstance(introduced_date, str):
        from datetime import datetime
        introduced_date = datetime.strptime(introduced_date, "%Y-%m-%d").date()

    # --- Bill type ---
    bill_type_raw = bill_row.get("bill_type", "")
    # Extract alpha prefix (e.g., "HB123" -> "HB", "HB" -> "HB")
    import re
    bill_type = re.match(r'^([A-Za-z]+)', bill_type_raw).group(1).upper() if re.match(r'^([A-Za-z]+)', bill_type_raw) else ""
    features["bill_type_encoded"] = BILL_TYPE_MAP.get(bill_type, 0)
    features["is_resolution"] = int(bill_type in ("HR", "SR", "HCR", "SCR", "HJR", "SJR"))
    features["is_joint_resolution"] = int(bill_type in ("HJR", "SJR"))
    features["originating_chamber"] = 1 if bill_row.get("body") == "H" else 0

    # --- Progress / status ---
    features["progress"] = bill_row.get("progress", 0)
    features["status"] = bill_row.get("status", 0)

    # --- Lifecycle / momentum ---
    if introduced_date and ref_date:
        features["days_since_introduction"] = max(0, (ref_date - introduced_date).days)
    else:
        features["days_since_introduction"] = 0

    last_action_date = bill_row.get("last_action_date")
    if isinstance(last_action_date, str):
        from datetime import datetime
        last_action_date = datetime.strptime(last_action_date, "%Y-%m-%d").date()

    if last_action_date and ref_date:
        features["days_since_last_action"] = max(0, (ref_date - last_action_date).days)
    else:
        features["days_since_last_action"] = 0

    # Number of history events (velocity proxy)
    history_count = db.execute(
        select(func.count(BillHistory.id))
        .where(BillHistory.bill_id == bill_id)
    ).scalar() or 0
    features["history_event_count"] = history_count

    # Early introduction advantage
    features["early_introduction"] = 0
    if introduced_date:
        # Approximate session start as January 1 of the odd year
        session_start_year = bill_row.get("year_start")
        if session_start_year:
            session_start = date(session_start_year, 1, 1)
        else:
            session_start = date(introduced_date.year, 1, 1)
        days_from_start = (introduced_date - session_start).days
        features["early_introduction"] = int(days_from_start <= EARLY_INTRODUCTION_DAYS)

    # Has received floor votes
    roll_call_count = db.execute(
        select(func.count(RollCall.id))
        .where(RollCall.bill_id == bill_id)
    ).scalar() or 0
    features["has_floor_vote"] = int(roll_call_count > 0)
    features["roll_call_count"] = roll_call_count

    # Successful roll calls (where passed=True)
    passed_roll_calls = db.execute(
        select(func.count(RollCall.id))
        .where(RollCall.bill_id == bill_id, RollCall.passed == True)  # noqa: E712
    ).scalar() or 0
    features["passed_roll_call_count"] = passed_roll_calls
    features["roll_call_success_rate"] = (
        passed_roll_calls / roll_call_count if roll_call_count > 0 else 0.0
    )

    # --- Amendments ---
    amendment_count = db.execute(
        select(func.count(Amendment.id))
        .where(Amendment.bill_id == bill_id)
    ).scalar() or 0
    features["amendment_count"] = amendment_count

    adopted_amendments = db.execute(
        select(func.count(Amendment.id))
        .where(Amendment.bill_id == bill_id, Amendment.adopted == True)  # noqa: E712
    ).scalar() or 0
    features["adopted_amendment_count"] = adopted_amendments

    # --- Content features ---
    features["text_length"] = bill_row.get("text_length", 0) or 0

    # Subject areas
    subjects_json = bill_row.get("subject_areas", "[]")
    try:
        subjects = json.loads(subjects_json) if isinstance(subjects_json, str) else []
    except json.JSONDecodeError:
        subjects = []
    features["num_subjects"] = len(subjects)

    # Budget/appropriations signal
    title = (bill_row.get("title") or "").lower()
    description = (bill_row.get("description") or "").lower()
    combined_text = f"{title} {description}"
    features["is_appropriations"] = int(
        any(kw in combined_text for kw in ["appropriat", "budget", "fiscal"])
    )

    # --- Companion bill detection ---
    features["has_companion"] = _detect_companion(db, bill_row)

    return features


def _detect_companion(db: DbSession, bill_row: dict) -> int:
    """Detect if a companion bill exists in the other chamber.

    A companion bill has the same or very similar title, introduced in
    the same session but in the opposite chamber.
    """
    session_id = bill_row.get("session_id")
    body = bill_row.get("body")
    title = bill_row.get("title", "")
    bill_id = bill_row.get("bill_id")

    if not session_id or not body or not title:
        return 0

    other_chamber = "S" if body == "H" else "H"

    # Look for bills in the other chamber with matching title
    matches = db.execute(
        select(func.count(Bill.bill_id))
        .where(
            Bill.session_id == session_id,
            Bill.body == other_chamber,
            Bill.title == title,
            Bill.bill_id != bill_id,
        )
    ).scalar() or 0

    return int(matches > 0)
