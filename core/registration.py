from dataclasses import dataclass

import numpy as np

from core.face_embedding import extract_embedding
from database.db import (
    DEFAULT_DB_PATH, DatabaseError, DuplicateStudentError, Student,
    get_student, initialize_database, insert_student,
)


@dataclass(frozen=True)
class RegistrationResult:
    success: bool
    code: str
    message: str
    student: Student | None = None


def register_student(student_id, full_name, frame, db_path=DEFAULT_DB_PATH):
    """Register one face from a captured uint8 BGR frame, without any GUI work.

    Creates the database if needed. Returns a result with code 'success',
    'invalid_input', 'duplicate_student', 'extraction_error', or 'database_error'.
    Face-count validation is delegated to extract_embedding. This synchronous
    function should run off the GUI thread when integrated with Tkinter later.
    """
    if (not isinstance(student_id, str) or not student_id.strip()
            or student_id.strip() == "Unknown"):
        return RegistrationResult(False, "invalid_input", "Enter a student ID other than 'Unknown'.")
    if not isinstance(full_name, str) or not full_name.strip():
        return RegistrationResult(False, "invalid_input", "Enter the student's full name.")
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3
            or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8):
        return RegistrationResult(False, "invalid_input", "Provide a captured uint8 BGR camera frame.")
    student_id, full_name = student_id.strip(), full_name.strip()

    try:
        initialize_database(db_path)
        if get_student(student_id, db_path) is not None:
            return RegistrationResult(False, "duplicate_student", f"Student ID {student_id!r} is already registered.")
    except DatabaseError as exc:
        return RegistrationResult(False, "database_error", str(exc))

    try:
        embedding = extract_embedding(frame)
    except Exception as exc:
        # Model/import/runtime failures are also actionable results for the future GUI.
        return RegistrationResult(False, "extraction_error", f"Embedding extraction failed: {exc}")

    try:
        student = insert_student(student_id, full_name, embedding, db_path)
    except DuplicateStudentError as exc:
        # The SQLite constraint also protects against inserts racing the earlier check.
        return RegistrationResult(False, "duplicate_student", str(exc))
    except DatabaseError as exc:
        return RegistrationResult(False, "database_error", str(exc))
    except ValueError as exc:
        return RegistrationResult(False, "extraction_error", str(exc))
    return RegistrationResult(True, "success", f"Student {student_id!r} registered successfully.", student)
