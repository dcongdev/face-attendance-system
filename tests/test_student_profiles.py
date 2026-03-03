from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from core.live_recognition import recognize_frame
from database.db import (
    DatabaseError, StudentNotFoundError, get_attendance_history, get_student,
    initialize_database, insert_student, record_attendance, search_students, update_student_name,
)


class StudentProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'students.sqlite3'
        initialize_database(self.path)
        self.embedding = np.linspace(-1, 1, 128, dtype=np.float32)
        insert_student('S01', 'Nguyễn An', self.embedding, self.path)
        insert_student('S02', 'Binh', -self.embedding, self.path)

    def test_search_by_id_name_and_unicode_casefold(self):
        for query, expected in [('s01', ['S01']), (' NGUYỄN ', ['S01']), ('BINH', ['S02']),
                                ('S0', ['S01','S02']), ('', ['S01','S02']), ('missing', [])]:
            with self.subTest(query=query):
                self.assertEqual([p.student_id for p in search_students(query, self.path)], expected)
        self.assertFalse(hasattr(search_students('', self.path)[0], 'embedding'))

    def test_search_is_literal_and_validates_query(self):
        insert_student('S%_', "O'Brien", self.embedding, self.path)
        self.assertEqual([p.student_id for p in search_students('%_', self.path)], ['S%_'])
        self.assertEqual(search_students("' OR 1=1 --", self.path), [])
        for query in (None, 1, []):
            with self.assertRaises(ValueError):
                search_students(query, self.path)

    def test_name_update_preserves_id_embedding_timestamp_and_attendance(self):
        before = get_student('S01', self.path)
        record_attendance('S01', self.path, now=datetime(2026,10,16,tzinfo=timezone.utc))
        updated = update_student_name(' S01 ', ' Nguyễn Ánh ', self.path)
        after = get_student('S01', self.path)
        self.assertEqual(updated.full_name, 'Nguyễn Ánh')
        self.assertEqual(after.student_id, before.student_id)
        self.assertEqual(after.registered_at, before.registered_at)
        np.testing.assert_array_equal(after.embedding, before.embedding)
        self.assertEqual(get_attendance_history(self.path)[0].full_name, 'Nguyễn Ánh')
        self.assertEqual(get_student('S02', self.path).full_name, 'Binh')

    def test_invalid_names_and_missing_students_do_not_update(self):
        for name in (None, '', '  ', 123):
            with self.assertRaises(ValueError):
                update_student_name('S01', name, self.path)
        for student_id in (None, '', 'Unknown'):
            with self.assertRaises(ValueError):
                update_student_name(student_id, 'An', self.path)
        with self.assertRaises(StudentNotFoundError):
            update_student_name('missing', 'An', self.path)
        self.assertEqual(get_student('S01', self.path).full_name, 'Nguyễn An')

    def test_name_update_uses_sql_parameters(self):
        name = "O'Brien'; DROP TABLE students; --"
        update_student_name('S01', name, self.path)
        self.assertEqual(get_student('S01', self.path).full_name, name)
        self.assertEqual(len(search_students('', self.path)), 2)

    def test_metadata_operations_do_not_decode_embeddings(self):
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute("UPDATE students SET embedding = 'bad' WHERE student_id = 'S01'")
        self.assertEqual(search_students('s01', self.path)[0].full_name, 'Nguyễn An')
        self.assertEqual(update_student_name('S01','An',self.path).full_name, 'An')

    def test_live_recognition_reads_updated_names_on_next_pass(self):
        frame = np.zeros((100,100,3), dtype=np.uint8)
        with patch('core.live_recognition.FaceDetector') as detector, patch(
            'core.live_recognition.extract_embedding', return_value=self.embedding
        ):
            detector.return_value.detect_faces.return_value = [(0,0,100,100)]
            self.assertEqual(recognize_frame(frame, db_path=self.path)[0][0].label, 'Nguyễn An (S01)')
            update_student_name('S01', 'New Name', self.path)
            self.assertEqual(recognize_frame(frame, db_path=self.path)[0][0].label, 'New Name (S01)')

    def test_database_errors_and_empty_list(self):
        missing = self.path.parent / 'missing.sqlite3'
        with self.assertRaises(DatabaseError):
            search_students('', missing)
        with self.assertRaises(DatabaseError):
            update_student_name('S01','An',missing)
        empty = self.path.parent / 'empty.sqlite3'
        initialize_database(empty)
        self.assertEqual(search_students('',empty), [])
