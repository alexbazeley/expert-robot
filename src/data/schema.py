"""SQLAlchemy ORM schema for the legislative bill database.

Tables capture the full lifecycle of bills: metadata, sponsors, cosponsors,
committee referrals, history events, roll calls, individual votes, and
legislator profiles. Designed for state-agnostic use but populated initially
with Ohio data.
"""

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from src.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Session(Base):
    """A legislative session (e.g., Ohio 136th General Assembly)."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, unique=True, nullable=False, index=True)
    state_abbr = Column(String(2), nullable=False)
    session_number = Column(Integer, nullable=False)  # e.g. 136
    session_name = Column(String(200), nullable=False)
    year_start = Column(Integer, nullable=False)
    year_end = Column(Integer, nullable=False)
    special = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    bills = relationship("Bill", back_populates="session")


class Legislator(Base):
    """A legislator (state representative or senator)."""

    __tablename__ = "legislators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    people_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    party = Column(String(50))
    party_id = Column(Integer)  # LegiScan party_id: 1=D, 2=R
    role = Column(String(100))  # "Rep" or "Sen"
    district = Column(String(50))
    state_abbr = Column(String(2))
    committee_sponsor = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sponsorships = relationship("BillSponsor", back_populates="legislator")
    votes = relationship("Vote", back_populates="legislator")


class Committee(Base):
    """A legislative committee."""

    __tablename__ = "committees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    committee_id = Column(Integer, unique=True, nullable=False, index=True)
    chamber = Column(String(10))  # "H" or "S"
    name = Column(String(300), nullable=False)
    state_abbr = Column(String(2))
    created_at = Column(DateTime, default=datetime.utcnow)

    referrals = relationship("CommitteeReferral", back_populates="committee")


class Bill(Base):
    """A legislative bill."""

    __tablename__ = "bills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bill_id = Column(Integer, unique=True, nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("sessions.session_id"), nullable=False, index=True)
    bill_number = Column(String(50), nullable=False)  # e.g. "HB 123"
    bill_type = Column(String(10))  # HB, SB, HJR, etc.
    bill_type_id = Column(Integer)
    title = Column(Text)
    description = Column(Text)
    state_abbr = Column(String(2), nullable=False)
    body = Column(String(10))  # "H" or "S" — originating chamber
    status = Column(Integer)  # LegiScan status code: 1-6
    status_label = Column(String(50))
    progress = Column(Integer, default=0)  # LegiScan progress: 0-4
    url = Column(Text)
    state_link = Column(Text)
    change_hash = Column(String(64))
    introduced_date = Column(Date)
    last_action_date = Column(Date)
    last_action = Column(Text)
    subject_areas = Column(Text)  # JSON-encoded list of subjects
    text_length = Column(Integer)  # character count of bill text
    enacted = Column(Boolean, default=False)  # derived: True if status==4
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session = relationship("Session", back_populates="bills")
    sponsors = relationship("BillSponsor", back_populates="bill")
    history = relationship("BillHistory", back_populates="bill", order_by="BillHistory.action_date")
    referrals = relationship("CommitteeReferral", back_populates="bill")
    roll_calls = relationship("RollCall", back_populates="bill")
    amendments = relationship("Amendment", back_populates="bill")

    __table_args__ = (
        Index("ix_bills_session_type", "session_id", "bill_type"),
        Index("ix_bills_session_status", "session_id", "status"),
    )


class BillSponsor(Base):
    """A sponsor or cosponsor of a bill."""

    __tablename__ = "bill_sponsors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bill_id = Column(Integer, ForeignKey("bills.bill_id"), nullable=False, index=True)
    people_id = Column(Integer, ForeignKey("legislators.people_id"), nullable=False, index=True)
    sponsor_type = Column(Integer)  # 1=primary sponsor, 0=cosponsor
    sponsor_order = Column(Integer)  # ordering from LegiScan

    bill = relationship("Bill", back_populates="sponsors")
    legislator = relationship("Legislator", back_populates="sponsorships")

    __table_args__ = (
        UniqueConstraint("bill_id", "people_id", name="uq_bill_sponsor"),
    )


class BillHistory(Base):
    """A status/action event in a bill's lifecycle."""

    __tablename__ = "bill_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bill_id = Column(Integer, ForeignKey("bills.bill_id"), nullable=False, index=True)
    action_date = Column(Date, nullable=False)
    action = Column(Text, nullable=False)
    chamber = Column(String(10))  # "H" or "S"
    importance = Column(Integer, default=0)  # LegiScan importance score
    sequence = Column(Integer)  # ordering

    bill = relationship("Bill", back_populates="history")

    __table_args__ = (
        Index("ix_history_bill_date", "bill_id", "action_date"),
    )


class CommitteeReferral(Base):
    """A bill's referral to a committee."""

    __tablename__ = "committee_referrals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bill_id = Column(Integer, ForeignKey("bills.bill_id"), nullable=False, index=True)
    committee_id = Column(Integer, ForeignKey("committees.committee_id"), nullable=False)
    referral_date = Column(Date)

    bill = relationship("Bill", back_populates="referrals")
    committee = relationship("Committee", back_populates="referrals")

    __table_args__ = (
        UniqueConstraint("bill_id", "committee_id", name="uq_bill_committee"),
    )


class RollCall(Base):
    """A roll call vote on a bill."""

    __tablename__ = "roll_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    roll_call_id = Column(Integer, unique=True, nullable=False, index=True)
    bill_id = Column(Integer, ForeignKey("bills.bill_id"), nullable=False, index=True)
    vote_date = Column(Date, nullable=False)
    chamber = Column(String(10))  # "H" or "S"
    description = Column(Text)
    yea = Column(Integer, default=0)
    nay = Column(Integer, default=0)
    nv = Column(Integer, default=0)  # not voting
    absent = Column(Integer, default=0)
    passed = Column(Boolean)

    bill = relationship("Bill", back_populates="roll_calls")
    votes = relationship("Vote", back_populates="roll_call")


class Vote(Base):
    """An individual legislator's vote on a roll call."""

    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    roll_call_id = Column(Integer, ForeignKey("roll_calls.roll_call_id"), nullable=False)
    people_id = Column(Integer, ForeignKey("legislators.people_id"), nullable=False)
    vote_value = Column(Integer)  # 1=Yea, 2=Nay, 3=NV, 4=Absent

    roll_call = relationship("RollCall", back_populates="votes")
    legislator = relationship("Legislator", back_populates="votes")

    __table_args__ = (
        Index("ix_votes_rollcall_person", "roll_call_id", "people_id"),
    )


class Amendment(Base):
    """An amendment to a bill."""

    __tablename__ = "amendments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    amendment_id = Column(Integer, unique=True, nullable=False)
    bill_id = Column(Integer, ForeignKey("bills.bill_id"), nullable=False, index=True)
    chamber = Column(String(10))
    title = Column(Text)
    description = Column(Text)
    adopted = Column(Boolean)
    amendment_date = Column(Date)

    bill = relationship("Bill", back_populates="amendments")


def get_engine(database_url: str | None = None):
    """Create and return a SQLAlchemy engine."""
    url = database_url or DATABASE_URL
    return create_engine(url, echo=False)


def get_session_factory(database_url: str | None = None):
    """Create and return a SQLAlchemy session factory."""
    engine = get_engine(database_url)
    return sessionmaker(bind=engine)


def init_database(database_url: str | None = None) -> None:
    """Create all tables in the database."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
