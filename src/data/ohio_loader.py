"""Ohio historical data loader.

Downloads bulk datasets from LegiScan for Ohio legislative sessions,
extracts JSON payloads from ZIP archives, and normalizes data into the
local SQLite database. Supports incremental sync via change_hash.

Usage:
    from src.data.ohio_loader import OhioDataLoader
    loader = OhioDataLoader()
    loader.load_all_sessions()
"""

import base64
import io
import json
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from src.config import (
    OHIO_SESSIONS_OF_INTEREST,
    OHIO_SESSION_YEARS,
    OHIO_STATE_ABBREVIATION,
    RAW_DATA_DIR,
    STATUS_CODES,
)
from src.data.legiscan_client import LegiScanClient
from src.data.schema import (
    Amendment,
    Bill,
    BillHistory,
    BillSponsor,
    Committee,
    CommitteeReferral,
    Legislator,
    RollCall,
    Session,
    Vote,
    get_session_factory,
    init_database,
)

logger = logging.getLogger(__name__)


def _parse_date(date_str: str | None) -> date | None:
    """Parse a date string from LegiScan (YYYY-MM-DD format)."""
    if not date_str or date_str == "0000-00-00":
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Failed to parse date: %s", date_str)
        return None


def _extract_bill_type(bill_number: str) -> str:
    """Extract bill type prefix from bill number (e.g., 'HB 123' -> 'HB', 'HB123' -> 'HB')."""
    import re
    match = re.match(r'^([A-Za-z]+)', bill_number.strip())
    return match.group(1).upper() if match else ""


def _ensure_list(value: Any) -> list:
    """Normalize a value that may be a dict (keyed by index) or a list.

    LegiScan bulk data sometimes encodes arrays as dicts with string keys
    like {"0": {...}, "1": {...}}. This normalizes to a list.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def _normalize_progress(progress_raw: Any) -> int:
    """Normalize the progress field from LegiScan data.

    In the per-bill API response, progress is an integer (0-4).
    In bulk dataset JSON, it may be a list of dicts like:
        [{'date': '2015-01-28', 'event': 1}, {'date': '2015-02-10', 'event': 2}]
    We extract the max event value as the current progress stage.
    """
    if isinstance(progress_raw, int):
        return progress_raw
    if isinstance(progress_raw, list):
        if not progress_raw:
            return 0
        # Each entry has an 'event' key with the progress stage
        events = [
            entry.get("event", entry.get("step", 0))
            for entry in progress_raw
            if isinstance(entry, dict)
        ]
        return max(events) if events else 0
    return 0


class OhioDataLoader:
    """Loads Ohio legislative data from LegiScan into the local database.

    Supports two modes:
    1. Bulk loading from ZIP dataset archives (for historical sessions)
    2. Incremental per-bill fetching (for current session updates)

    Args:
        client: LegiScanClient instance. Created with defaults if not provided.
        database_url: SQLAlchemy database URL. Uses default if not provided.
    """

    def __init__(
        self,
        client: LegiScanClient | None = None,
        database_url: str | None = None,
    ) -> None:
        self.client = client or LegiScanClient()
        self.state = OHIO_STATE_ABBREVIATION
        init_database(database_url)
        self._session_factory = get_session_factory(database_url)
        self._bulk_data_dir = RAW_DATA_DIR / "bulk_datasets"
        self._bulk_data_dir.mkdir(parents=True, exist_ok=True)

    def load_all_sessions(
        self,
        sessions: list[int] | None = None,
        force_reload: bool = False,
    ) -> None:
        """Load data for all Ohio sessions of interest.

        Args:
            sessions: Specific session numbers to load. Defaults to all.
            force_reload: If True, re-download and re-load even if data exists.
        """
        target_sessions = sessions or OHIO_SESSIONS_OF_INTEREST
        session_list = self.client.get_session_list(self.state)

        # Map session numbers to LegiScan session IDs
        session_map = self._map_sessions(session_list, target_sessions)

        for session_num, session_info in sorted(session_map.items()):
            session_id = session_info["session_id"]
            logger.info(
                "Loading session %d (%s), LegiScan ID: %d",
                session_num,
                session_info["session_name"],
                session_id,
            )
            self._ensure_session_record(session_info, session_num)
            self._load_session_data(session_id, session_num, force_reload)

    def _map_sessions(
        self,
        session_list: list[dict],
        target_sessions: list[int],
    ) -> dict[int, dict]:
        """Map target session numbers to LegiScan session info dicts.

        Matches by year range from OHIO_SESSION_YEARS config.

        Args:
            session_list: Raw session list from LegiScan API.
            target_sessions: Session numbers we want (e.g., [131, 132, ...]).

        Returns:
            Dict mapping session_number -> session_info dict.
        """
        session_map: dict[int, dict] = {}
        for session_num in target_sessions:
            years = OHIO_SESSION_YEARS.get(session_num)
            if not years:
                logger.warning("No year range configured for session %d", session_num)
                continue
            for s in session_list:
                if s.get("year_start") == years[0] and s.get("year_end") == years[1]:
                    if not s.get("special", 0):
                        session_map[session_num] = s
                        break
            else:
                logger.warning(
                    "Could not find LegiScan session for Ohio %dth GA (%d-%d)",
                    session_num,
                    years[0],
                    years[1],
                )
        return session_map

    def _ensure_session_record(self, session_info: dict, session_num: int) -> None:
        """Insert or update the session record in the database."""
        with self._session_factory() as db:
            existing = db.execute(
                select(Session).where(Session.session_id == session_info["session_id"])
            ).scalar_one_or_none()

            if existing is None:
                years = OHIO_SESSION_YEARS.get(session_num, (0, 0))
                session_record = Session(
                    session_id=session_info["session_id"],
                    state_abbr=self.state,
                    session_number=session_num,
                    session_name=session_info.get("session_name", f"{session_num}th GA"),
                    year_start=years[0],
                    year_end=years[1],
                    special=bool(session_info.get("special", 0)),
                )
                db.add(session_record)
                db.commit()
                logger.info("Created session record for %dth GA", session_num)

    def _load_session_data(
        self,
        session_id: int,
        session_num: int,
        force_reload: bool,
    ) -> None:
        """Load bill data for a session, preferring bulk datasets.

        Tries to download and extract the bulk ZIP dataset first. Falls
        back to per-bill API fetching if bulk data is unavailable.
        """
        # Try bulk dataset first
        bulk_dir = self._download_bulk_dataset(session_id)
        if bulk_dir:
            self._load_from_bulk(bulk_dir, session_id, session_num)
        else:
            logger.info("No bulk dataset, falling back to per-bill fetch for session %d", session_id)
            self._load_from_api(session_id, session_num)

    def _download_bulk_dataset(self, session_id: int) -> Path | None:
        """Download and extract a bulk dataset ZIP for a session.

        Returns:
            Path to extracted directory, or None if no dataset available.
        """
        extract_dir = self._bulk_data_dir / str(session_id)
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info("Bulk dataset already extracted for session %d", session_id)
            return extract_dir

        try:
            datasets = self.client.get_dataset_list(state=self.state)
        except Exception as e:
            logger.warning("Failed to get dataset list: %s", e)
            return None

        # Find the matching dataset
        target_dataset = None
        for ds in datasets:
            if ds.get("session_id") == session_id:
                target_dataset = ds
                break

        if not target_dataset:
            logger.info("No bulk dataset available for session %d", session_id)
            return None

        access_key = target_dataset.get("access_key", "")
        if not access_key:
            logger.warning("No access_key for session %d dataset", session_id)
            return None

        try:
            logger.info("Downloading bulk dataset for session %d...", session_id)
            result = self.client.get_dataset(session_id, access_key)
            zip_data = base64.b64decode(result.get("zip", ""))
        except Exception as e:
            logger.warning("Failed to download dataset for session %d: %s", session_id, e)
            return None

        # Extract ZIP
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                zf.extractall(extract_dir)
            logger.info("Extracted bulk dataset to %s", extract_dir)
            return extract_dir
        except zipfile.BadZipFile as e:
            logger.warning("Bad ZIP file for session %d: %s", session_id, e)
            return None

    def _load_from_bulk(self, bulk_dir: Path, session_id: int, session_num: int) -> None:
        """Load session data from extracted bulk dataset JSON files."""
        # Find bill JSON files (typically in a subdirectory like /bill/)
        bill_files = list(bulk_dir.rglob("bill/*.json"))
        if not bill_files:
            # Some datasets have flat structure
            bill_files = [f for f in bulk_dir.rglob("*.json") if "bill" in f.stem.lower()]

        logger.info("Loading %d bill files from bulk dataset", len(bill_files))

        # Also load person data
        person_files = list(bulk_dir.rglob("people/*.json"))
        if not person_files:
            person_files = list(bulk_dir.rglob("person/*.json"))

        with self._session_factory() as db:
            # Load legislators first
            for pf in person_files:
                try:
                    with open(pf) as f:
                        data = json.load(f)
                    person_data = data.get("person", data)
                    self._upsert_legislator(db, person_data)
                except Exception as e:
                    db.rollback()
                    logger.warning("Failed to load person file %s: %s", pf, e)

            try:
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning("Failed to commit legislators: %s", e)

            # Load bills — commit in batches with per-bill error isolation
            loaded = 0
            failed = 0
            for bf in bill_files:
                try:
                    with open(bf) as f:
                        data = json.load(f)
                    bill_data = data.get("bill", data)
                    self._upsert_bill(db, bill_data, session_id, session_num)
                    loaded += 1

                    # Commit in batches of 100 to avoid holding too much in memory
                    if loaded % 100 == 0:
                        db.commit()
                        logger.info("  Committed %d bills...", loaded)

                except Exception as e:
                    db.rollback()
                    failed += 1
                    if failed <= 5:
                        logger.warning("Failed to load bill file %s: %s", bf.name, e)
                    elif failed == 6:
                        logger.warning("Suppressing further bill load warnings...")

            try:
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning("Failed to commit final bill batch: %s", e)

            logger.info(
                "Bulk data for session %d: %d loaded, %d failed",
                session_id, loaded, failed,
            )

        # Load roll calls
        rollcall_files = list(bulk_dir.rglob("roll_call/*.json"))
        if rollcall_files:
            rc_loaded = 0
            with self._session_factory() as db:
                for rf in rollcall_files:
                    try:
                        with open(rf) as f:
                            data = json.load(f)
                        rc_data = data.get("roll_call", data)
                        self._upsert_roll_call(db, rc_data)
                        rc_loaded += 1
                        if rc_loaded % 100 == 0:
                            db.commit()
                    except Exception as e:
                        db.rollback()
                        logger.debug("Failed to load roll call file %s: %s", rf.name, e)
                try:
                    db.commit()
                except Exception as e:
                    db.rollback()
            logger.info("Loaded %d/%d roll call files for session %d", rc_loaded, len(rollcall_files), session_id)

    def _load_from_api(self, session_id: int, session_num: int) -> None:
        """Load session data by fetching each bill individually from the API."""
        master = self.client.get_master_list(session_id)

        with self._session_factory() as db:
            for key, bill_summary in master.items():
                bill_id = bill_summary.get("bill_id")
                if not bill_id:
                    continue

                try:
                    bill_data = self.client.get_bill(bill_id)
                    self._upsert_bill(db, bill_data, session_id, session_num)

                    # Fetch roll calls
                    for vote_info in bill_data.get("votes", []):
                        rc_id = vote_info.get("roll_call_id")
                        if rc_id:
                            try:
                                rc_data = self.client.get_roll_call(rc_id)
                                self._upsert_roll_call(db, rc_data)
                            except Exception as e:
                                logger.warning("Failed to fetch roll call %d: %s", rc_id, e)

                    # Fetch sponsor/legislator details
                    for sponsor in bill_data.get("sponsors", []):
                        pid = sponsor.get("people_id")
                        if pid:
                            try:
                                person = self.client.get_person(pid)
                                self._upsert_legislator(db, person)
                            except Exception as e:
                                logger.warning("Failed to fetch person %d: %s", pid, e)

                except Exception as e:
                    logger.warning("Failed to fetch bill %d: %s", bill_id, e)

            db.commit()

    def _upsert_legislator(self, db: DbSession, person: dict) -> None:
        """Insert or update a legislator record."""
        people_id = person.get("people_id")
        if not people_id:
            return

        existing = db.execute(
            select(Legislator).where(Legislator.people_id == people_id)
        ).scalar_one_or_none()

        if existing:
            existing.name = person.get("name", existing.name)
            existing.first_name = person.get("first_name", existing.first_name)
            existing.last_name = person.get("last_name", existing.last_name)
            existing.party = person.get("party", existing.party)
            existing.party_id = person.get("party_id", existing.party_id)
            existing.role = person.get("role", existing.role)
            existing.district = person.get("district", existing.district)
            existing.state_abbr = self.state
        else:
            db.add(Legislator(
                people_id=people_id,
                name=person.get("name", ""),
                first_name=person.get("first_name"),
                last_name=person.get("last_name"),
                party=person.get("party"),
                party_id=person.get("party_id"),
                role=person.get("role"),
                district=person.get("district"),
                state_abbr=self.state,
            ))

    def _upsert_bill(
        self,
        db: DbSession,
        bill_data: dict,
        session_id: int,
        session_num: int,
    ) -> None:
        """Insert or update a bill and its related records."""
        bill_id = bill_data.get("bill_id")
        if not bill_id:
            return

        bill_number = bill_data.get("bill_number", "")
        bill_type = _extract_bill_type(bill_number)
        status_raw = bill_data.get("status", 0)
        status = status_raw if isinstance(status_raw, int) else 0
        progress = _normalize_progress(bill_data.get("progress", 0))

        # Parse introduced date from history
        introduced_date = None
        history = _ensure_list(bill_data.get("history", []))
        if history and isinstance(history[0], dict):
            introduced_date = _parse_date(history[0].get("date"))

        last_action_date = _parse_date(bill_data.get("last_action_date"))
        last_action = bill_data.get("last_action", "")

        # Subjects — handle both list-of-dicts and list-of-strings
        subjects_raw = _ensure_list(bill_data.get("subjects", []))
        subject_names = []
        for s in (subjects_raw or []):
            if isinstance(s, dict):
                subject_names.append(s.get("subject_name", ""))
            elif isinstance(s, str):
                subject_names.append(s)
        subject_text = json.dumps(subject_names)

        # Text length (from texts list if available)
        text_length = 0
        texts = _ensure_list(bill_data.get("texts", []))
        if texts:
            # Use doc_size of the most recent text version
            text_entries = [t for t in texts if isinstance(t, dict)]
            if text_entries:
                latest_text = max(text_entries, key=lambda t: t.get("date", ""), default={})
                text_length = latest_text.get("doc_size", 0)

        existing = db.execute(
            select(Bill).where(Bill.bill_id == bill_id)
        ).scalar_one_or_none()

        if existing:
            existing.status = status
            existing.status_label = STATUS_CODES.get(status, "")
            existing.progress = progress
            existing.last_action_date = last_action_date
            existing.last_action = last_action
            existing.change_hash = bill_data.get("change_hash")
            existing.enacted = status == 4
            existing.text_length = text_length
            existing.subject_areas = subject_text
        else:
            bill_record = Bill(
                bill_id=bill_id,
                session_id=session_id,
                bill_number=bill_number,
                bill_type=bill_type,
                bill_type_id=bill_data.get("bill_type_id"),
                title=bill_data.get("title"),
                description=bill_data.get("description"),
                state_abbr=self.state,
                body=bill_data.get("body"),
                status=status,
                status_label=STATUS_CODES.get(status, ""),
                progress=progress,
                url=bill_data.get("url"),
                state_link=bill_data.get("state_link"),
                change_hash=bill_data.get("change_hash"),
                introduced_date=introduced_date,
                last_action_date=last_action_date,
                last_action=last_action,
                subject_areas=subject_text,
                text_length=text_length,
                enacted=status == 4,
            )
            db.add(bill_record)

        # Sponsors
        self._load_sponsors(db, bill_id, _ensure_list(bill_data.get("sponsors", [])))

        # History
        self._load_history(db, bill_id, _ensure_list(history))

        # Committee referrals
        self._load_committees(db, bill_id, bill_data.get("committee", {}))

        # Amendments
        self._load_amendments(db, bill_id, _ensure_list(bill_data.get("amendments", [])))

    def _load_sponsors(
        self,
        db: DbSession,
        bill_id: int,
        sponsors: list[dict],
    ) -> None:
        """Load sponsor/cosponsor records for a bill."""
        for i, sponsor in enumerate(sponsors):
            people_id = sponsor.get("people_id")
            if not people_id:
                continue

            # Ensure legislator exists
            existing_leg = db.execute(
                select(Legislator).where(Legislator.people_id == people_id)
            ).scalar_one_or_none()
            if not existing_leg:
                db.add(Legislator(
                    people_id=people_id,
                    name=sponsor.get("name", ""),
                    first_name=sponsor.get("first_name"),
                    last_name=sponsor.get("last_name"),
                    party=sponsor.get("party"),
                    party_id=sponsor.get("party_id"),
                    role=sponsor.get("role"),
                    district=sponsor.get("district"),
                    state_abbr=self.state,
                ))

            existing = db.execute(
                select(BillSponsor).where(
                    BillSponsor.bill_id == bill_id,
                    BillSponsor.people_id == people_id,
                )
            ).scalar_one_or_none()

            if not existing:
                db.add(BillSponsor(
                    bill_id=bill_id,
                    people_id=people_id,
                    sponsor_type=sponsor.get("sponsor_type_id", 0),
                    sponsor_order=sponsor.get("sponsor_order", i),
                ))

    def _load_history(
        self,
        db: DbSession,
        bill_id: int,
        history: list[dict],
    ) -> None:
        """Load bill history/action events."""
        # Clear existing history for this bill to avoid duplicates on reload
        db.query(BillHistory).filter(BillHistory.bill_id == bill_id).delete()

        for i, event in enumerate(history):
            action_date = _parse_date(event.get("date"))
            if not action_date:
                continue
            db.add(BillHistory(
                bill_id=bill_id,
                action_date=action_date,
                action=event.get("action", ""),
                chamber=event.get("chamber"),
                importance=event.get("importance", 0),
                sequence=i,
            ))

    def _load_committees(self, db: DbSession, bill_id: int, committee: dict) -> None:
        """Load committee referral for a bill."""
        if not committee:
            return

        # committee can be a single dict or a list
        committees = [committee] if isinstance(committee, dict) else committee

        for comm in committees:
            committee_id = comm.get("committee_id")
            if not committee_id:
                continue

            # Ensure committee record exists
            existing_comm = db.execute(
                select(Committee).where(Committee.committee_id == committee_id)
            ).scalar_one_or_none()
            if not existing_comm:
                db.add(Committee(
                    committee_id=committee_id,
                    chamber=comm.get("chamber"),
                    name=comm.get("name", ""),
                    state_abbr=self.state,
                ))

            # Add referral if not exists
            existing_ref = db.execute(
                select(CommitteeReferral).where(
                    CommitteeReferral.bill_id == bill_id,
                    CommitteeReferral.committee_id == committee_id,
                )
            ).scalar_one_or_none()
            if not existing_ref:
                db.add(CommitteeReferral(
                    bill_id=bill_id,
                    committee_id=committee_id,
                ))

    def _load_amendments(
        self,
        db: DbSession,
        bill_id: int,
        amendments: list[dict],
    ) -> None:
        """Load amendment records for a bill."""
        for amend in amendments:
            amendment_id = amend.get("amendment_id")
            if not amendment_id:
                continue

            existing = db.execute(
                select(Amendment).where(Amendment.amendment_id == amendment_id)
            ).scalar_one_or_none()

            if not existing:
                db.add(Amendment(
                    amendment_id=amendment_id,
                    bill_id=bill_id,
                    chamber=amend.get("chamber"),
                    title=amend.get("title"),
                    description=amend.get("description"),
                    adopted=amend.get("adopted"),
                    amendment_date=_parse_date(amend.get("date")),
                ))

    def _upsert_roll_call(self, db: DbSession, rc_data: dict) -> None:
        """Insert or update a roll call and its individual votes."""
        roll_call_id = rc_data.get("roll_call_id")
        bill_id = rc_data.get("bill_id")
        if not roll_call_id or not bill_id:
            return

        # Ensure the bill exists before linking
        bill_exists = db.execute(
            select(Bill.bill_id).where(Bill.bill_id == bill_id)
        ).scalar_one_or_none()
        if not bill_exists:
            return

        existing = db.execute(
            select(RollCall).where(RollCall.roll_call_id == roll_call_id)
        ).scalar_one_or_none()

        if not existing:
            db.add(RollCall(
                roll_call_id=roll_call_id,
                bill_id=bill_id,
                vote_date=_parse_date(rc_data.get("date")),
                chamber=rc_data.get("chamber"),
                description=rc_data.get("desc"),
                yea=rc_data.get("yea", 0),
                nay=rc_data.get("nay", 0),
                nv=rc_data.get("nv", 0),
                absent=rc_data.get("absent", 0),
                passed=rc_data.get("passed"),
            ))

            # Individual votes
            for vote in _ensure_list(rc_data.get("votes", [])):
                people_id = vote.get("people_id")
                if not people_id:
                    continue
                db.add(Vote(
                    roll_call_id=roll_call_id,
                    people_id=people_id,
                    vote_value=vote.get("vote_id"),
                ))

    def sync_current_session(self, session_id: int, session_num: int) -> int:
        """Incrementally sync the current session using change_hash.

        Only re-fetches bills whose change_hash has changed since the
        last sync, minimizing API calls.

        Args:
            session_id: LegiScan session ID.
            session_num: Ohio GA session number.

        Returns:
            Number of bills updated.
        """
        # Get known hashes from database
        with self._session_factory() as db:
            bills = db.execute(
                select(Bill.bill_id, Bill.change_hash).where(Bill.session_id == session_id)
            ).all()
            known_hashes = {b.bill_id: b.change_hash for b in bills}

        changed_ids = self.client.get_changed_bills(session_id, known_hashes)

        if not changed_ids:
            logger.info("No bill changes detected for session %d", session_id)
            return 0

        logger.info("Syncing %d changed bills for session %d", len(changed_ids), session_id)

        with self._session_factory() as db:
            for bill_id in changed_ids:
                try:
                    bill_data = self.client.get_bill(bill_id)
                    self._upsert_bill(db, bill_data, session_id, session_num)
                except Exception as e:
                    logger.warning("Failed to sync bill %d: %s", bill_id, e)
            db.commit()

        return len(changed_ids)
