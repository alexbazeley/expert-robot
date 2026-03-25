"""Single bill passage prediction with interpretable output.

Fetches current bill data, computes features, runs the trained model,
and produces a prediction with SHAP-based factor attribution, confidence
context, base rate comparison, and a plain-English narrative summary.

Usage:
    from src.predict import predict_bill
    result = predict_bill(bill_id=1234567)
    print(result["narrative"])
"""

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.config import OHIO_SESSION_YEARS, OHIO_STATE_ABBREVIATION
from src.data.legiscan_client import LegiScanClient
from src.data.ohio_loader import OhioDataLoader
from src.data.schema import Bill, Session, get_session_factory, init_database
from src.features.build_features import (
    FEATURE_COLUMNS,
    build_single_bill_features,
    compute_bill_features_all,
)
from src.models.evaluate import get_bill_explanation
from src.models.passage_model import PassageModel

logger = logging.getLogger(__name__)

# Human-readable feature names for output
FEATURE_LABELS: dict[str, str] = {
    "sponsor_majority_party": "Sponsor is in majority party",
    "sponsor_leadership": "Sponsor holds leadership position",
    "sponsor_seniority": "Sponsor seniority (prior sessions)",
    "sponsor_success_rate": "Sponsor's historical success rate",
    "sponsor_bills_this_session": "Bills sponsor introduced this session",
    "sponsor_is_chair_of_committee": "Sponsor chairs assigned committee",
    "cosponsor_count": "Number of cosponsors",
    "total_sponsor_count": "Total sponsors and cosponsors",
    "bipartisan_cosponsor_score": "Bipartisan cosponsorship",
    "cosponsors_on_committee": "Cosponsors on assigned committee",
    "cosponsor_chair_on_committee": "Committee chair is cosponsor",
    "cross_chamber_cosponsors": "Cosponsors from other chamber",
    "committee_pass_through_rate": "Committee historical pass-through rate",
    "committee_hearing_count": "Committee actions on this bill",
    "committee_chair_is_sponsor": "Committee chair sponsors this bill",
    "progress": "Current progress stage",
    "status": "Current status code",
    "days_since_introduction": "Days since introduction",
    "days_since_last_action": "Days since last action (staleness)",
    "history_event_count": "Number of status events",
    "early_introduction": "Introduced early in session",
    "has_floor_vote": "Has received floor vote(s)",
    "roll_call_count": "Number of roll call votes",
    "roll_call_success_rate": "Roll call success rate",
    "amendment_count": "Number of amendments",
    "has_companion": "Companion bill in other chamber",
    "bill_type_encoded": "Bill type",
    "is_resolution": "Is a resolution",
    "text_length": "Bill text length",
    "is_appropriations": "Appropriations/budget bill",
    "session_pct_elapsed": "Session % elapsed",
    "total_session_bills": "Total bills this session",
    "base_rate_bill_type": "Historical base rate (bill type)",
    "base_rate_committee": "Historical base rate (committee)",
    "trifecta": "Unified government trifecta",
    "supermajority": "Supermajority control",
}

PROGRESS_LABELS: dict[int, str] = {
    0: "Prefiled",
    1: "Introduced",
    2: "Passed Committee / Engrossed",
    3: "Passed One Chamber / Enrolled",
    4: "Passed Both Chambers",
}


def predict_bill(
    bill_id: int | None = None,
    state: str = "OH",
    bill_number: str | None = None,
    model_type: str = "xgboost",
    database_url: str | None = None,
) -> dict[str, Any]:
    """Generate a full prediction with explanation for a single bill.

    Either bill_id or bill_number must be provided. If bill_number is
    provided, the bill is looked up via LegiScan search.

    Args:
        bill_id: LegiScan bill ID.
        state: State abbreviation (default "OH").
        bill_number: Bill number like "HB 123" (alternative to bill_id).
        model_type: Model type to use ("xgboost" or "logistic").
        database_url: Override database URL.

    Returns:
        Prediction result dict with probability, explanation, and narrative.
    """
    if bill_id is None and bill_number is None:
        raise ValueError("Must provide either bill_id or bill_number")

    client = LegiScanClient()

    # Resolve bill_number to bill_id if needed
    if bill_id is None and bill_number is not None:
        bill_id = _resolve_bill_number(client, state, bill_number)

    # Fetch fresh bill data and ensure it's in the database
    logger.info("Fetching bill %d from LegiScan...", bill_id)
    bill_data = client.get_bill(bill_id)

    # Ensure bill is in database for feature computation
    loader = OhioDataLoader(client=client, database_url=database_url)
    init_database(database_url)
    session_factory = get_session_factory(database_url)

    with session_factory() as db:
        from sqlalchemy import select
        existing = db.execute(
            select(Bill).where(Bill.bill_id == bill_id)
        ).scalar_one_or_none()

        if not existing:
            # Need to load this bill into the database
            session_id = bill_data.get("session", {}).get("session_id")
            if session_id:
                session_record = db.execute(
                    select(Session).where(Session.session_id == session_id)
                ).scalar_one_or_none()
                session_num = session_record.session_number if session_record else 136
                loader._upsert_bill(db, bill_data, session_id, session_num)
                db.commit()

    # Compute features
    snapshot = date.today()
    features = build_single_bill_features(bill_id, database_url, snapshot)

    # Load trained model
    model = PassageModel.load(state="ohio", model_type=model_type)

    # Get prediction with SHAP explanation
    explanation = get_bill_explanation(model, features)

    # Construct result
    result: dict[str, Any] = {
        "bill_id": bill_id,
        "bill_number": bill_data.get("bill_number", features.get("bill_number", "")),
        "title": bill_data.get("title", ""),
        "state": state,
        "status": bill_data.get("status", 0),
        "status_label": bill_data.get("status_desc", ""),
        "progress": bill_data.get("progress", 0),
        "progress_label": PROGRESS_LABELS.get(bill_data.get("progress", 0), "Unknown"),
        "prediction": {
            "p_enacted": explanation["p_enacted"],
            "p_committee": explanation["p_committee"],
            "p_enacted_given_committee": explanation["p_enacted_given_committee"],
        },
        "base_rate": features.get("base_rate_bill_type", 0.0),
        "relative_likelihood": (
            explanation["p_enacted"] / features["base_rate_bill_type"]
            if features.get("base_rate_bill_type", 0) > 0
            else None
        ),
        "top_positive_factors": [
            {
                "feature": FEATURE_LABELS.get(f["feature"], f["feature"]),
                "feature_key": f["feature"],
                "impact": f["impact"],
                "value": f["value"],
            }
            for f in explanation["positive_factors"]
        ],
        "top_negative_factors": [
            {
                "feature": FEATURE_LABELS.get(f["feature"], f["feature"]),
                "feature_key": f["feature"],
                "impact": f["impact"],
                "value": f["value"],
            }
            for f in explanation["negative_factors"]
        ],
        "features": {k: v for k, v in features.items() if k in FEATURE_COLUMNS},
        "snapshot_date": snapshot.isoformat(),
    }

    # Generate narrative
    result["narrative"] = _generate_narrative(result)

    return result


def _resolve_bill_number(client: LegiScanClient, state: str, bill_number: str) -> int:
    """Search for a bill by number and return its bill_id.

    Args:
        client: LegiScanClient instance.
        state: State abbreviation.
        bill_number: Bill number like "HB 123".

    Returns:
        LegiScan bill_id.

    Raises:
        ValueError: If the bill cannot be found.
    """
    results = client.search(state=state, query=bill_number)

    # Filter results for exact match
    for key, result in results.items():
        if key == "summary":
            continue
        if isinstance(result, dict):
            result_number = result.get("bill_number", "")
            if result_number.replace(" ", "") == bill_number.replace(" ", ""):
                return result["bill_id"]

    # If no exact match, try the first result
    for key, result in results.items():
        if key == "summary":
            continue
        if isinstance(result, dict) and "bill_id" in result:
            logger.warning(
                "No exact match for '%s', using closest: %s",
                bill_number,
                result.get("bill_number"),
            )
            return result["bill_id"]

    raise ValueError(f"Could not find bill '{bill_number}' in {state}")


def _generate_narrative(result: dict[str, Any]) -> str:
    """Generate a plain-English narrative summary of the prediction.

    Written for a policy professional audience — factual, calibrated
    in its language, and explicit about uncertainty.
    """
    bill = result["bill_number"]
    title = result.get("title", "")
    p = result["prediction"]["p_enacted"]
    p_committee = result["prediction"]["p_committee"]
    base_rate = result.get("base_rate", 0)
    relative = result.get("relative_likelihood")
    progress_label = result.get("progress_label", "")

    lines = []

    # Header
    lines.append(f"## Prediction: {bill}")
    if title:
        lines.append(f"**{title}**\n")

    # Current status
    lines.append(f"**Current stage:** {progress_label}")

    # Probability
    pct = p * 100
    if pct < 5:
        strength = "Very unlikely"
    elif pct < 15:
        strength = "Unlikely"
    elif pct < 30:
        strength = "Possible but below average"
    elif pct < 50:
        strength = "Moderate chance"
    elif pct < 70:
        strength = "Likely"
    else:
        strength = "Very likely"

    lines.append(f"\n**Estimated probability of enactment: {pct:.1f}%** ({strength})")
    lines.append(f"- Probability of exiting committee: {p_committee * 100:.1f}%")
    lines.append(
        f"- Probability of enactment given committee passage: "
        f"{result['prediction']['p_enacted_given_committee'] * 100:.1f}%"
    )

    # Base rate context
    if base_rate > 0:
        lines.append(
            f"\n**Base rate context:** The historical passage rate for this bill type "
            f"is {base_rate * 100:.1f}%."
        )
        if relative is not None:
            if relative > 1.5:
                lines.append(
                    f"This bill is **{relative:.1f}x more likely** to pass than "
                    f"the average bill of its type."
                )
            elif relative < 0.67:
                lines.append(
                    f"This bill is **{1/relative:.1f}x less likely** to pass than "
                    f"the average bill of its type."
                )
            else:
                lines.append("This bill's chances are roughly in line with the average.")

    # Top factors
    pos = result.get("top_positive_factors", [])
    neg = result.get("top_negative_factors", [])

    if pos:
        lines.append("\n**Factors favoring passage:**")
        for f in pos[:5]:
            lines.append(f"- {f['feature']}")

    if neg:
        lines.append("\n**Factors working against passage:**")
        for f in neg[:5]:
            lines.append(f"- {f['feature']}")

    # Caveats
    lines.append("\n---")
    lines.append("*This is a statistical estimate based on historical patterns. "
                 "It cannot account for political negotiations, external events, "
                 "or gubernatorial decisions not yet reflected in the legislative record. "
                 "Predictions are point-in-time and should be updated as the bill "
                 "accumulates new actions.*")

    return "\n".join(lines)
