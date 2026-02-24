from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from database.db import (
    DatabaseError, DuplicateStudentError, get_student, initialize_database,
    insert_student, load_registered_students,
)


class StudentDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data" / "students.sqlite3"
        self.embedding = np.linspace(-1, 1, 128, dtype=np.float32)
        initialize_database(self.path)

    def test_round_trip_and_timestamp(self):
        before = datetime.now(timezone.utc)
        saved = insert_student(" S01 ", " Nguyễn Văn An ", self.embedding, self.path)
        loaded = get_student("S01", self.path)
        self.assertEqual(loaded.student_id, "S01")
        self.assertEqual(loaded.full_name, "Nguyễn Văn An")
        self.assertEqual(loaded.registered_at, saved.registered_at)
        timestamp = datetime.fromisoformat(loaded.registered_at)
        self.assertLessEqual(before, timestamp)
        self.assertLessEqual(timestamp, datetime.now(timezone.utc))
        self.assertEqual(loaded.embedding.dtype, np.float32)
        np.testing.assert_array_equal(loaded.embedding, self.embedding)

    def test_empty_and_missing(self):
        self.assertEqual(load_registered_students(self.path), {})
        self.assertIsNone(get_student("missing", self.path))

    def test_initialize_is_idempotent(self):
        insert_student("S01", "An", self.embedding.tolist(), self.path)
        initialize_database(self.path)
        self.assertIsNotNone(get_student("S01", self.path))

    def test_duplicate_preserves_original_and_database_remains_usable(self):
        insert_student("S01", "An", self.embedding, self.path)
        with self.assertRaises(DuplicateStudentError):
            insert_student(" S01 ", "Other", -self.embedding, self.path)
        self.assertEqual(get_student("S01", self.path).full_name, "An")
        insert_student("S02", "Binh", self.embedding, self.path)
        self.assertEqual(len(load_registered_students(self.path)), 2)

    def test_load_matching_mapping_and_independent_arrays(self):
        for student_id in ("S02", "S01"):
            insert_student(student_id, "Name", self.embedding, self.path)
        known = load_registered_students(self.path)
        self.assertEqual(list(known), ["S01", "S02"])
        np.testing.assert_array_equal(known["S01"], self.embedding)
        known["S01"][:] = 0
        np.testing.assert_array_equal(get_student("S01", self.path).embedding, self.embedding)

    def test_invalid_embeddings_do_not_write(self):
        for value in (None, [], [1]*127, [[1]*128], [0]*128,
                      [np.nan]*128, [np.inf]*128, [1e300]*128, ["1"]*128, [1j]*128):
            with self.subTest(value=str(value)[:40]), self.assertRaises(ValueError):
                insert_student("S01", "Name", value, self.path)
        self.assertEqual(load_registered_students(self.path), {})

    def test_invalid_student_fields(self):
        for student_id, name in [(None, "Name"), (" ", "Name"), ("Unknown", "Name"),
                                 (12, "Name"), ("S01", None), ("S01", " ")]:
            with self.subTest(student_id=student_id, name=name), self.assertRaises(ValueError):
                insert_student(student_id, name, self.embedding, self.path)

    def test_sql_parameters_preserve_literal_values(self):
        student_id = "S01'; DROP TABLE students;--"
        insert_student(student_id, "O'Brien", self.embedding, self.path)
        self.assertEqual(get_student(student_id, self.path).full_name, "O'Brien")
        self.assertEqual(len(load_registered_students(self.path)), 1)

    def test_database_errors(self):
        missing = self.path.parent / "missing.sqlite3"
        with self.assertRaises(DatabaseError):
            get_student("S01", missing)
        self.assertFalse(missing.exists())
        invalid = self.path.parent / "broken.sqlite3"
        invalid.write_text("not a database")
        with self.assertRaises(DatabaseError):
            initialize_database(invalid)
        with self.assertRaises(DatabaseError):
            initialize_database(self.path / "child.sqlite3")

    def test_corrupt_embedding_is_reported(self):
        insert_student("S01", "Name", self.embedding, self.path)
        for value in ("bad json", "[1,2]", "null"):
            with closing(sqlite3.connect(self.path)) as connection:
                with connection:
                    connection.execute("UPDATE students SET embedding = ?", (value,))
            for operation in (lambda: get_student("S01", self.path),
                              lambda: load_registered_students(self.path)):
                with self.subTest(value=value), self.assertRaises(DatabaseError):
                    operation()


if __name__ == "__main__":
    unittest.main()
