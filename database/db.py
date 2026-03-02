from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "attendance.sqlite3"
ATTENDANCE_TIMEZONE = timezone(timedelta(hours=7), name="Asia/Saigon")


class DatabaseError(Exception):
    """A database operation failed or stored data is invalid."""


class DuplicateStudentError(DatabaseError):
    """The student ID is already registered."""


class StudentNotFoundError(DatabaseError):
    """Attendance requires an existing registered student."""


@dataclass
class Student:
    student_id: str
    full_name: str
    embedding: np.ndarray
    registered_at: str


@dataclass(frozen=True)
class AttendanceRecord:
    student_id: str
    full_name: str
    attendance_date: str
    checked_in_at: str
    record_id: int


@contextmanager
def _connection(db_path, *, create=False):
    connection = None
    try:
        path = Path(db_path).resolve()
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path.as_uri() + ("?mode=rwc" if create else "?mode=rw"), uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection
    except (sqlite3.Error, OSError) as exc:
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()


def initialize_database(db_path=DEFAULT_DB_PATH):
    """Create local storage and student/attendance tables, safely on repeated calls."""
    with _connection(db_path, create=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY NOT NULL,
                full_name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                registered_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(student_id),
                attendance_date TEXT NOT NULL,
                checked_in_at TEXT NOT NULL,
                UNIQUE (student_id, attendance_date)
            )
        """)
        if "id" not in {row["name"] for row in connection.execute("PRAGMA table_info(attendance)")}:
            # Upgrade the original composite-key table transactionally, preserving all check-ins.
            connection.execute("""
                CREATE TABLE attendance_migration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    attendance_date TEXT NOT NULL,
                    checked_in_at TEXT NOT NULL,
                    UNIQUE (student_id, attendance_date)
                )
            """)
            connection.execute("""
                INSERT INTO attendance_migration (student_id, attendance_date, checked_in_at)
                SELECT student_id, attendance_date, checked_in_at FROM attendance
            """)
            connection.execute("DROP TABLE attendance")
            connection.execute("ALTER TABLE attendance_migration RENAME TO attendance")


def _student_id(value):
    if not isinstance(value, str) or not value.strip() or value.strip() == "Unknown":
        raise ValueError("Student ID must be a nonempty string other than 'Unknown'.")
    return value.strip()


def _embedding(value):
    try:
        array = np.asarray(value)
        if array.shape != (128,) or array.dtype.kind not in "iuf":
            raise ValueError("Expected 128 real numbers.")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32)
        if not np.isfinite(array).all() or not np.any(array):
            raise ValueError("Expected a finite, nonzero vector.")
        return array
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Embedding must be a finite, nonzero 128-dimensional FaceNet vector.") from exc


def insert_student(student_id, full_name, embedding, db_path=DEFAULT_DB_PATH):
    """Insert and return a Student; duplicate IDs never overwrite existing data.

    IDs/names are stripped. Invalid input raises ValueError. Embeddings are
    serialized as JSON numbers and restored as float32, matching FaceNet output.
    Registration timestamps are timezone-aware UTC ISO 8601 strings.
    """
    student_id = _student_id(student_id)
    if not isinstance(full_name, str) or not full_name.strip():
        raise ValueError("Full name must be a nonempty string.")
    student = Student(student_id, full_name.strip(), _embedding(embedding),
                      datetime.now(timezone.utc).isoformat())
    with _connection(db_path) as connection:
        try:
            connection.execute(
                "INSERT INTO students (student_id, full_name, embedding, registered_at) VALUES (?, ?, ?, ?)",
                (student.student_id, student.full_name,
                 json.dumps(student.embedding.tolist(), allow_nan=False), student.registered_at),
            )
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorcode in (sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY, sqlite3.SQLITE_CONSTRAINT_UNIQUE):
                raise DuplicateStudentError(f"Student ID {student_id!r} is already registered.") from exc
            raise
    return student


def _restore(row):
    try:
        return Student(row["student_id"], row["full_name"],
                       _embedding(json.loads(row["embedding"])), row["registered_at"])
    except (ValueError, TypeError) as exc:
        raise DatabaseError(f"Invalid stored embedding for student {row['student_id']!r}.") from exc


def get_student(student_id, db_path=DEFAULT_DB_PATH):
    """Return a Student with a NumPy embedding, or None when the ID is absent."""
    student_id = _student_id(student_id)
    with _connection(db_path) as connection:
        row = connection.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    return _restore(row) if row is not None else None


def load_registered_students(db_path=DEFAULT_DB_PATH):
    """Return {student_id: float32_embedding} for the independent matching helper."""
    return {student.student_id: student.embedding for student in list_students(db_path)}


def list_students(db_path=DEFAULT_DB_PATH):
    """Load student records and embeddings together from one database snapshot."""
    with _connection(db_path) as connection:
        rows = connection.execute("SELECT * FROM students ORDER BY student_id").fetchall()
    return [_restore(row) for row in rows]


def _attendance_time(now):
    if now is None:
        now = datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Check-in time must be a timezone-aware datetime.")
    return (now.astimezone(ATTENDANCE_TIMEZONE).date().isoformat(),
            now.astimezone(timezone.utc).isoformat())


def _require_student(connection, student_id):
    if connection.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,)).fetchone() is None:
        raise StudentNotFoundError(f"Student ID {student_id!r} is not registered.")


def record_attendance(student_id, db_path=DEFAULT_DB_PATH, *, now=None, cancel_event=None):
    """Return True for a new check-in, False for an existing check-in that day.

    Days use Asia/Saigon (UTC+07); timestamps are stored as UTC ISO 8601 strings.
    Optional aware `now` permits deterministic callers/tests. Duplicate check-ins
    retain the original timestamp. Raise ValueError for malformed IDs/times,
    StudentNotFoundError for absent students, or DatabaseError on storage failure.
    Call initialize_database before using attendance functions.
    Optional cancel_event cancels a queued/in-progress write before commit,
    returning None. An already committed check-in is retained.
    """
    student_id = _student_id(student_id)
    attendance_date, checked_in_at = _attendance_time(now)
    with _connection(db_path) as connection:
        # Acquire the SQLite write lock before checking cancellation: a write
        # that waited for another connection must not proceed after Stop.
        connection.execute("BEGIN IMMEDIATE")
        if cancel_event is not None and cancel_event.is_set():
            return None
        _require_student(connection, student_id)
        cursor = connection.execute("""
            INSERT INTO attendance (student_id, attendance_date, checked_in_at)
            VALUES (?, ?, ?)
            ON CONFLICT(student_id, attendance_date) DO NOTHING
        """, (student_id, attendance_date, checked_in_at))
        if cancel_event is not None and cancel_event.is_set():
            connection.rollback()
            return None
        return cursor.rowcount == 1


def has_checked_in_today(student_id, db_path=DEFAULT_DB_PATH, *, now=None):
    """Check the Asia/Saigon calendar day; invalid/missing IDs follow record_attendance."""
    student_id = _student_id(student_id)
    attendance_date, _ = _attendance_time(now)
    with _connection(db_path) as connection:
        _require_student(connection, student_id)
        return connection.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND attendance_date = ?",
            (student_id, attendance_date),
        ).fetchone() is not None


def get_attendance_history(db_path=DEFAULT_DB_PATH):
    """Return attendance with student names, newest first; ties sort by student ID."""
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT a.id AS record_id, a.student_id, s.full_name, a.attendance_date, a.checked_in_at
            FROM attendance AS a JOIN students AS s ON s.student_id = a.student_id
            ORDER BY a.checked_in_at DESC, a.student_id ASC
        """).fetchall()
    return [AttendanceRecord(**dict(row)) for row in rows]


def delete_attendance_records(record_ids, db_path=DEFAULT_DB_PATH):
    """Delete only explicit stable attendance IDs; return the number removed.

    Empty input is a no-op, absent IDs are ignored, and invalid IDs raise
    ValueError before any write. Student profiles and embeddings are untouched.
    """
    try:
        ids = list(record_ids)
    except TypeError as exc:
        raise ValueError("Record IDs must be a collection of positive integers.") from exc
    if any(type(record_id) is not int or record_id <= 0 for record_id in ids):
        raise ValueError("Record IDs must be positive integers.")
    ids = sorted(set(ids))
    if not ids:
        return 0
    with _connection(db_path) as connection:
        deleted = 0
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            deleted += connection.execute(
                f"DELETE FROM attendance WHERE id IN ({placeholders})", batch
            ).rowcount
        return deleted
