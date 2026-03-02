from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from database.db import (
    DatabaseError, delete_attendance_records, get_attendance_history, get_student,
    initialize_database, insert_student, record_attendance,
)


class AttendanceDeletionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'attendance.sqlite3'
        initialize_database(self.path)
        self.embedding = np.ones(128, dtype=np.float32)
        insert_student('S01', 'An', self.embedding, self.path)
        self.now = datetime(2026,10,16,3,tzinfo=timezone.utc)
        for offset in range(3):
            record_attendance('S01', self.path, now=self.now - timedelta(days=offset))

    def test_selected_ids_delete_only_those_records_and_preserve_students(self):
        records = get_attendance_history(self.path)
        ids = [records[0].record_id, records[2].record_id]
        self.assertEqual(delete_attendance_records(ids + [ids[0]], self.path), 2)
        self.assertEqual([row.record_id for row in get_attendance_history(self.path)], [records[1].record_id])
        student = get_student('S01', self.path)
        self.assertEqual(student.full_name, 'An')
        np.testing.assert_array_equal(student.embedding, self.embedding)

    def test_empty_missing_and_invalid_ids(self):
        self.assertEqual(delete_attendance_records([], self.path), 0)
        self.assertEqual(delete_attendance_records([99999], self.path), 0)
        valid_id = get_attendance_history(self.path)[0].record_id
        for ids in (None, '1', [True], [0], [-1], [1.5], [valid_id, '1 OR 1=1']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                delete_attendance_records(ids, self.path)
        self.assertEqual(len(get_attendance_history(self.path)), 3)

    def test_clear_filter_and_all_using_explicit_ids(self):
        records = get_attendance_history(self.path)
        ids = [row.record_id for row in records if row.attendance_date == '2026-10-15']
        self.assertEqual(delete_attendance_records(ids, self.path), 1)
        self.assertEqual(delete_attendance_records([r.record_id for r in get_attendance_history(self.path)], self.path), 2)
        self.assertEqual(get_attendance_history(self.path), [])
        self.assertIsNotNone(get_student('S01', self.path))

    def test_ids_remain_stable_and_are_never_reused(self):
        records = get_attendance_history(self.path)
        ids = [row.record_id for row in records]
        initialize_database(self.path)
        self.assertEqual([row.record_id for row in get_attendance_history(self.path)], ids)
        delete_attendance_records(ids, self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('VACUUM')
        record_attendance('S01', self.path, now=self.now)
        self.assertGreater(get_attendance_history(self.path)[0].record_id, max(ids))
        self.assertEqual(delete_attendance_records(ids, self.path), 0)
        self.assertEqual(len(get_attendance_history(self.path)), 1)

    def test_original_schema_migration_preserves_checkins_and_constraints(self):
        before = get_attendance_history(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('DROP TABLE attendance')
                connection.execute('''CREATE TABLE attendance (
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    attendance_date TEXT NOT NULL, checked_in_at TEXT NOT NULL,
                    PRIMARY KEY (student_id, attendance_date))''')
                connection.executemany('INSERT INTO attendance VALUES (?, ?, ?)',
                                       [(r.student_id,r.attendance_date,r.checked_in_at) for r in before])
        initialize_database(self.path)
        migrated = get_attendance_history(self.path)
        self.assertEqual([(r.student_id,r.attendance_date,r.checked_in_at) for r in migrated],
                         [(r.student_id,r.attendance_date,r.checked_in_at) for r in before])
        self.assertEqual(len({r.record_id for r in migrated}), 3)
        self.assertFalse(record_attendance('S01', self.path, now=self.now))
        self.assertIsNotNone(get_student('S01', self.path))

    def test_database_error_does_not_delete_any_records(self):
        records = get_attendance_history(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('''CREATE TRIGGER block_delete BEFORE DELETE ON attendance
                                  BEGIN SELECT RAISE(ABORT, 'blocked'); END''')
        with self.assertRaises(DatabaseError):
            delete_attendance_records([r.record_id for r in records], self.path)
        self.assertEqual(len(get_attendance_history(self.path)), 3)
        with self.assertRaises(DatabaseError):
            delete_attendance_records([1], self.path.parent / 'missing.sqlite3')

    def test_large_batch_is_atomic(self):
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.executemany('INSERT INTO attendance (student_id,attendance_date,checked_in_at) VALUES (?,?,?)',
                                       [('S01', (self.now-timedelta(days=10+i)).date().isoformat(), self.now.isoformat()) for i in range(501)])
                last_id = connection.execute('SELECT MAX(id) FROM attendance').fetchone()[0]
                connection.execute(f'''CREATE TRIGGER block_last BEFORE DELETE ON attendance WHEN OLD.id = {last_id}
                                       BEGIN SELECT RAISE(ABORT, 'blocked'); END''')
        records = get_attendance_history(self.path)
        with self.assertRaises(DatabaseError):
            delete_attendance_records([r.record_id for r in records], self.path)
        self.assertEqual(len(get_attendance_history(self.path)), len(records))
