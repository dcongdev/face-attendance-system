from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "attendance.sqlite3"


class DatabaseError(Exception):
    """A database operation failed or stored data is invalid."""


class DuplicateStudentError(DatabaseError):
    """The student ID is already registered."""


@dataclass
class Student:
    student_id: str
    full_name: str
    embedding: np.ndarray
    registered_at: str


@contextmanager
def _connection(db_path, *, create=False):
    connection = None
    try:
        path = Path(db_path).resolve()
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path.as_uri() + ("?mode=rwc" if create else "?mode=rw"), uri=True)
        connection.row_factory = sqlite3.Row
        with connection:
            yield connection
    except (sqlite3.Error, OSError) as exc:
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()


def initialize_database(db_path=DEFAULT_DB_PATH):
    """Create local storage and the students table, safely on repeated calls."""
    with _connection(db_path, create=True) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY NOT NULL,
                full_name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                registered_at TEXT NOT NULL
            )
        """)


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
