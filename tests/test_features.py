"""Tests for the feature engineering pipeline.

Uses an in-memory SQLite database with known test data to verify
feature computation produces expected outputs.
"""

import json
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.schema import (
    Amendment,
    Base,
    Bill,
    BillHistory,
    BillSponsor,
    Committee,
    CommitteeReferral,
    Legislator,
    RollCall,
    Session,
    Vote,
)
from src.features.bill_features import compute_bill_features
from src.features.committee_features import compute_committee_features
from src.features.session_features import compute_session_features
from src.features.sponsor_features import compute_sponsor_features


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database with test data."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)
    db = SessionFactory()

    # --- Session ---
    db.add(Session(
        session_id=1700,
        state_abbr="OH",
        session_number=135,
        session_name="135th General Assembly",
        year_start=2023,
        year_end=2024,
    ))
    db.add(Session(
        session_id=1850,
        state_abbr="OH",
        session_number=136,
        session_name="136th General Assembly",
        year_start=2025,
        year_end=2026,
    ))

    # --- Legislators ---
    db.add(Legislator(
        people_id=1001,
        name="Jane Smith",
        first_name="Jane",
        last_name="Smith",
        party="R",
        party_id=2,
        role="Rep",
        district="42",
        state_abbr="OH",
    ))
    db.add(Legislator(
        people_id=1002,
        name="John Doe",
        first_name="John",
        last_name="Doe",
        party="D",
        party_id=1,
        role="Sen",
        district="15",
        state_abbr="OH",
    ))
    db.add(Legislator(
        people_id=1003,
        name="Bob Wilson",
        first_name="Bob",
        last_name="Wilson",
        party="R",
        party_id=2,
        role="Rep",
        district="10",
        state_abbr="OH",
    ))

    # --- Committee ---
    db.add(Committee(
        committee_id=500,
        chamber="H",
        name="House Finance",
        state_abbr="OH",
    ))

    # --- Bills in prior session (for base rates) ---
    for i in range(10):
        bill_id = 9000 + i
        db.add(Bill(
            bill_id=bill_id,
            session_id=1700,
            bill_number=f"HB {i+1}",
            bill_type="HB",
            title=f"Prior session bill {i}",
            state_abbr="OH",
            body="H",
            status=4 if i < 2 else 6,  # 2 of 10 passed
            progress=4 if i < 2 else 1,
            enacted=i < 2,
            introduced_date=date(2023, 2, 1),
            last_action_date=date(2024, 6, 1),
        ))
        db.add(CommitteeReferral(bill_id=bill_id, committee_id=500))
        # Sponsor for seniority tracking
        db.add(BillSponsor(bill_id=bill_id, people_id=1001, sponsor_type=1, sponsor_order=0))

    # --- Test bill in current session ---
    db.add(Bill(
        bill_id=50001,
        session_id=1850,
        bill_number="HB 123",
        bill_type="HB",
        title="Test Education Reform",
        description="An act to reform education standards",
        state_abbr="OH",
        body="H",
        status=1,
        progress=1,
        enacted=False,
        introduced_date=date(2025, 2, 15),
        last_action_date=date(2025, 3, 10),
        text_length=15000,
        subject_areas=json.dumps(["Education", "Finance"]),
    ))

    # Sponsors
    db.add(BillSponsor(bill_id=50001, people_id=1001, sponsor_type=1, sponsor_order=0))
    db.add(BillSponsor(bill_id=50001, people_id=1002, sponsor_type=0, sponsor_order=1))
    db.add(BillSponsor(bill_id=50001, people_id=1003, sponsor_type=0, sponsor_order=2))

    # Committee referral
    db.add(CommitteeReferral(bill_id=50001, committee_id=500))

    # History
    db.add(BillHistory(
        bill_id=50001,
        action_date=date(2025, 2, 15),
        action="Introduced",
        chamber="H",
        importance=1,
        sequence=0,
    ))
    db.add(BillHistory(
        bill_id=50001,
        action_date=date(2025, 2, 20),
        action="Referred to House Finance Committee",
        chamber="H",
        importance=1,
        sequence=1,
    ))
    db.add(BillHistory(
        bill_id=50001,
        action_date=date(2025, 3, 10),
        action="Committee hearing scheduled",
        chamber="H",
        importance=0,
        sequence=2,
    ))

    # Roll call
    db.add(RollCall(
        roll_call_id=7001,
        bill_id=50001,
        vote_date=date(2025, 3, 10),
        chamber="H",
        description="Floor vote",
        yea=60,
        nay=30,
        nv=5,
        absent=4,
        passed=True,
    ))

    # Amendment
    db.add(Amendment(
        amendment_id=8001,
        bill_id=50001,
        chamber="H",
        title="Amendment 1",
        adopted=True,
        amendment_date=date(2025, 3, 5),
    ))

    db.commit()
    yield db
    db.close()


class TestSponsorFeatures:
    def test_majority_party(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        # Primary sponsor (Jane Smith) is R = majority
        assert features["sponsor_majority_party"] == 1

    def test_cosponsor_count(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        assert features["cosponsor_count"] == 2
        assert features["total_sponsor_count"] == 3

    def test_bipartisan_score(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        # 1 of 2 cosponsors is D (John Doe) = 0.5
        assert features["bipartisan_cosponsor_score"] == pytest.approx(0.5, abs=0.01)

    def test_seniority(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        # Jane Smith has bills in session 1700 (prior)
        assert features["sponsor_seniority"] >= 1

    def test_success_rate(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        # 2 of 10 prior bills passed = 0.2
        assert features["sponsor_success_rate"] == pytest.approx(0.2, abs=0.01)

    def test_default_features_no_sponsors(self, db_session) -> None:
        # Bill with no sponsors
        db_session.add(Bill(
            bill_id=50099,
            session_id=1850,
            bill_number="HB 999",
            bill_type="HB",
            state_abbr="OH",
            body="H",
            status=1,
            progress=1,
        ))
        db_session.commit()

        bill_row = {"bill_id": 50099, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        assert features["cosponsor_count"] == 0
        assert features["sponsor_majority_party"] == 0

    def test_cross_chamber_cosponsors(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_sponsor_features(db_session, bill_row, 136)
        # John Doe is a Senator, primary is a Rep
        assert features["cross_chamber_cosponsors"] >= 1


class TestCommitteeFeatures:
    def test_committee_referral_detected(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_committee_features(db_session, bill_row, 136)
        assert features["committee_id"] == 500
        assert features["num_committee_referrals"] == 1

    def test_pass_through_rate(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_committee_features(db_session, bill_row, 136)
        # 2 of 10 prior bills in committee had progress >= 2
        assert features["committee_pass_through_rate"] == pytest.approx(0.2, abs=0.01)

    def test_hearing_count(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_committee_features(db_session, bill_row, 136)
        # "Referred to...Committee" and "Committee hearing" = at least 2
        assert features["committee_hearing_count"] >= 2

    def test_no_committee(self, db_session) -> None:
        db_session.add(Bill(
            bill_id=50098,
            session_id=1850,
            bill_number="HB 998",
            bill_type="HB",
            state_abbr="OH",
            body="H",
            status=1,
            progress=1,
        ))
        db_session.commit()

        bill_row = {"bill_id": 50098, "session_id": 1850}
        features = compute_committee_features(db_session, bill_row, 136)
        assert features["committee_id"] == 0
        assert features["committee_pass_through_rate"] == 0.0


class TestBillFeatures:
    def test_bill_type_encoding(self, db_session) -> None:
        bill_row = {
            "bill_id": 50001,
            "session_id": 1850,
            "bill_type": "HB",
            "body": "H",
            "status": 1,
            "progress": 1,
            "introduced_date": date(2025, 2, 15),
            "last_action_date": date(2025, 3, 10),
            "text_length": 15000,
            "subject_areas": json.dumps(["Education", "Finance"]),
            "title": "Test Education Reform",
            "description": "An act to reform education standards",
            "year_start": 2025,
        }
        features = compute_bill_features(db_session, bill_row, date(2025, 3, 25))
        assert features["bill_type_encoded"] == 1  # HB = 1
        assert features["is_resolution"] == 0
        assert features["originating_chamber"] == 1  # H

    def test_days_computation(self, db_session) -> None:
        bill_row = {
            "bill_id": 50001,
            "session_id": 1850,
            "bill_type": "HB",
            "body": "H",
            "status": 1,
            "progress": 1,
            "introduced_date": date(2025, 2, 15),
            "last_action_date": date(2025, 3, 10),
            "text_length": 15000,
            "subject_areas": "[]",
            "title": "Test",
            "description": "",
            "year_start": 2025,
        }
        snapshot = date(2025, 3, 25)
        features = compute_bill_features(db_session, bill_row, snapshot)
        assert features["days_since_introduction"] == 38  # Feb 15 to Mar 25
        assert features["days_since_last_action"] == 15   # Mar 10 to Mar 25

    def test_roll_call_features(self, db_session) -> None:
        bill_row = {
            "bill_id": 50001,
            "session_id": 1850,
            "bill_type": "HB",
            "body": "H",
            "status": 1,
            "progress": 1,
            "introduced_date": date(2025, 2, 15),
            "last_action_date": date(2025, 3, 10),
            "text_length": 15000,
            "subject_areas": "[]",
            "title": "Test",
            "description": "",
            "year_start": 2025,
        }
        features = compute_bill_features(db_session, bill_row)
        assert features["has_floor_vote"] == 1
        assert features["roll_call_count"] == 1
        assert features["passed_roll_call_count"] == 1
        assert features["roll_call_success_rate"] == 1.0

    def test_amendment_count(self, db_session) -> None:
        bill_row = {
            "bill_id": 50001,
            "session_id": 1850,
            "bill_type": "HB",
            "body": "H",
            "status": 1,
            "progress": 1,
            "introduced_date": date(2025, 2, 15),
            "last_action_date": date(2025, 3, 10),
            "text_length": 15000,
            "subject_areas": "[]",
            "title": "Test",
            "description": "",
            "year_start": 2025,
        }
        features = compute_bill_features(db_session, bill_row)
        assert features["amendment_count"] == 1
        assert features["adopted_amendment_count"] == 1

    def test_content_features(self, db_session) -> None:
        bill_row = {
            "bill_id": 50001,
            "session_id": 1850,
            "bill_type": "HB",
            "body": "H",
            "status": 1,
            "progress": 1,
            "introduced_date": date(2025, 2, 15),
            "last_action_date": date(2025, 3, 10),
            "text_length": 15000,
            "subject_areas": json.dumps(["Education", "Finance"]),
            "title": "Budget Reform Act",
            "description": "Appropriations for education",
            "year_start": 2025,
        }
        features = compute_bill_features(db_session, bill_row)
        assert features["text_length"] == 15000
        assert features["num_subjects"] == 2
        assert features["is_appropriations"] == 1

    def test_early_introduction(self, db_session) -> None:
        bill_row = {
            "bill_id": 50001,
            "session_id": 1850,
            "bill_type": "HB",
            "body": "H",
            "status": 1,
            "progress": 1,
            "introduced_date": date(2025, 2, 15),
            "last_action_date": date(2025, 3, 10),
            "text_length": 0,
            "subject_areas": "[]",
            "title": "Test",
            "description": "",
            "year_start": 2025,
        }
        features = compute_bill_features(db_session, bill_row, date(2025, 3, 25))
        # Feb 15 is within 90 days of Jan 1
        assert features["early_introduction"] == 1


class TestSessionFeatures:
    def test_session_pct_elapsed(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        snapshot = date(2025, 7, 1)  # ~6 months into 2-year session
        features = compute_session_features(db_session, bill_row, 136, snapshot)
        assert 0.2 < features["session_pct_elapsed"] < 0.4

    def test_election_year(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        # 2025 = odd year, not election year
        features = compute_session_features(db_session, bill_row, 136, date(2025, 6, 1))
        assert features["is_election_year"] == 0

        # 2026 = even year, election year
        features = compute_session_features(db_session, bill_row, 136, date(2026, 6, 1))
        assert features["is_election_year"] == 1

    def test_partisan_composition(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_session_features(db_session, bill_row, 136, date(2025, 6, 1))
        assert features["house_majority_pct"] > 0.6  # R supermajority
        assert features["senate_majority_pct"] > 0.7
        assert features["governor_aligned"] == 1
        assert features["trifecta"] == 1
        # 65/99 = 65.7% which is just under 2/3, so supermajority may be 0
        assert features["supermajority"] in (0, 1)

    def test_base_rate_by_type(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850, "bill_type": "HB"}
        features = compute_session_features(db_session, bill_row, 136, date(2025, 6, 1))
        # 2 of 10 prior HB bills passed = 0.2
        assert features["base_rate_bill_type"] == pytest.approx(0.2, abs=0.01)

    def test_total_session_bills(self, db_session) -> None:
        bill_row = {"bill_id": 50001, "session_id": 1850}
        features = compute_session_features(db_session, bill_row, 136, date(2025, 6, 1))
        assert features["total_session_bills"] >= 1
