from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from database.db import (
    DatabaseError, StudentNotFoundError, get_attendance_history, get_student,
    has_checked_in_today, initialize_database, insert_student, record_attendance,
)


class AttendanceDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'attendance.sqlite3'
        initialize_database(self.path)
        insert_student('S01', 'Nguyễn An', np.ones(128), self.path)
        self.now = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)

    def test_record_and_history(self):
        self.assertFalse(has_checked_in_today('S01', self.path, now=self.now))
        self.assertTrue(record_attendance(' S01 ', self.path, now=self.now))
        self.assertTrue(has_checked_in_today('S01', self.path, now=self.now))
        record = get_attendance_history(self.path)[0]
        self.assertEqual(record.student_id, 'S01')
        self.assertEqual(record.full_name, 'Nguyễn An')
        self.assertEqual(record.attendance_date, '2026-10-16')
        self.assertEqual(datetime.fromisoformat(record.checked_in_at), self.now)

    def test_duplicate_retains_first_timestamp(self):
        self.assertTrue(record_attendance('S01', self.path, now=self.now))
        self.assertFalse(record_attendance('S01', self.path, now=self.now + timedelta(hours=1)))
        history = get_attendance_history(self.path)
        self.assertEqual(len(history), 1)
        self.assertEqual(datetime.fromisoformat(history[0].checked_in_at), self.now)

    def test_new_day_and_saigon_midnight(self):
        before_midnight = datetime(2026, 10, 16, 16, 59, tzinfo=timezone.utc)
        after_midnight = before_midnight + timedelta(minutes=1)
        self.assertTrue(record_attendance('S01', self.path, now=before_midnight))
        self.assertFalse(has_checked_in_today('S01', self.path, now=after_midnight))
        self.assertTrue(record_attendance('S01', self.path, now=after_midnight))
        self.assertEqual([r.attendance_date for r in get_attendance_history(self.path)],
                         ['2026-10-17', '2026-10-16'])

    def test_empty_history_and_independent_students(self):
        self.assertEqual(get_attendance_history(self.path), [])
        insert_student('S02', 'Binh', np.ones(128), self.path)
        for student_id in ('S02', 'S01'):
            self.assertTrue(record_attendance(student_id, self.path, now=self.now))
        self.assertEqual([r.student_id for r in get_attendance_history(self.path)], ['S01', 'S02'])

    def test_missing_and_invalid_student_ids(self):
        for operation in (record_attendance, has_checked_in_today):
            with self.subTest(operation=operation.__name__), self.assertRaises(StudentNotFoundError):
                operation('missing', self.path, now=self.now)
            for student_id in (None, '', ' ', 123, 'Unknown'):
                with self.subTest(student_id=student_id), self.assertRaises(ValueError):
                    operation(student_id, self.path, now=self.now)
        self.assertEqual(get_attendance_history(self.path), [])

    def test_invalid_timestamp(self):
        for now in ('2026-10-16', datetime(2026,10,16), 123):
            for operation in (record_attendance, has_checked_in_today):
                with self.subTest(now=now), self.assertRaises(ValueError):
                    operation('S01', self.path, now=now)

    def test_today_uses_current_clock_by_default(self):
        instant = self.now

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls.fromtimestamp(instant.timestamp(), tz)

        with patch('database.db.datetime', FixedDateTime):
            self.assertTrue(record_attendance('S01', self.path))
            self.assertTrue(has_checked_in_today('S01', self.path))

    def test_uniqueness_and_foreign_key_at_database_level(self):
        record_attendance('S01', self.path, now=self.now)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('PRAGMA foreign_keys = ON')
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute('INSERT INTO attendance (student_id, attendance_date, checked_in_at) VALUES (?, ?, ?)',
                                   ('S01','2026-10-16',self.now.isoformat()))
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute('INSERT INTO attendance (student_id, attendance_date, checked_in_at) VALUES (?, ?, ?)',
                                   ('missing','2026-10-16',self.now.isoformat()))

    def test_concurrent_checkins_insert_once(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: record_attendance('S01', self.path, now=self.now), range(2)))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(len(get_attendance_history(self.path)), 1)

    def test_existing_student_database_gets_attendance_table(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('DROP TABLE attendance')
        initialize_database(self.path)
        initialize_database(self.path)
        self.assertIsNotNone(get_student('S01', self.path))
        self.assertTrue(record_attendance('S01', self.path, now=self.now))

    def test_database_errors(self):
        missing = self.path.parent / 'missing.sqlite3'
        for operation in (lambda: record_attendance('S01', missing, now=self.now),
                          lambda: has_checked_in_today('S01', missing, now=self.now),
                          lambda: get_attendance_history(missing)):
            with self.assertRaises(DatabaseError):
                operation()
        self.assertFalse(missing.exists())
