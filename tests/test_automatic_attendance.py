from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from core.attendance import AttendanceSession
from core.live_recognition import RecognizedFace
from database.db import (
    DatabaseError, get_attendance_history, initialize_database, insert_student,
    record_attendance,
)


class AutomaticAttendanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'attendance.sqlite3'
        initialize_database(self.path)
        insert_student('S01', 'An', np.ones(128), self.path)
        self.session = AttendanceSession(self.path)
        self.face = RecognizedFace((0,0,100,100), 'An (S01)', 'S01')
        self.now = datetime(2026,10,16,3,tzinfo=timezone.utc)

    def test_success_and_repeated_faces_are_logged_and_reported_once(self):
        messages = self.session.check_in([self.face, self.face], now=self.now)
        self.assertEqual(messages, ['Checked in: An (S01)'])
        self.assertEqual(self.session.check_in([self.face], now=self.now), [])
        self.assertEqual(len(get_attendance_history(self.path)), 1)

    def test_new_session_reports_existing_checkin_once(self):
        self.session.check_in([self.face], now=self.now)
        session = AttendanceSession(self.path)
        self.assertEqual(session.check_in([self.face], now=self.now), ['Already checked in today: An (S01)'])
        self.assertEqual(session.check_in([self.face], now=self.now), [])
        self.assertEqual(len(get_attendance_history(self.path)), 1)

    def test_unknown_and_display_text_are_never_used_as_ids(self):
        faces = [RecognizedFace((0,0,1,1), 'Unknown'),
                 RecognizedFace((0,0,1,1), 'An (S01)'),
                 RecognizedFace((0,0,1,1), 'Unknown', 'Unknown')]
        self.assertEqual(self.session.check_in(faces, now=self.now), [])
        self.assertEqual(get_attendance_history(self.path), [])

    def test_new_day_logs_again(self):
        self.session.check_in([self.face], now=self.now)
        self.assertEqual(self.session.check_in([self.face], now=self.now + timedelta(days=1)),
                         ['Checked in: An (S01)'])
        self.assertEqual(len(get_attendance_history(self.path)), 2)

    def test_stop_prevents_new_checkins(self):
        self.session.stop()
        self.assertEqual(self.session.check_in([self.face], now=self.now), [])
        self.assertEqual(get_attendance_history(self.path), [])

    def test_errors_are_deduplicated_and_retried(self):
        with patch('core.attendance.record_attendance', side_effect=DatabaseError('locked')):
            self.assertIn('locked', self.session.check_in([self.face], now=self.now)[0])
            self.assertEqual(self.session.check_in([self.face], now=self.now), [])
        self.assertEqual(self.session.check_in([self.face], now=self.now), ['Checked in: An (S01)'])

    def test_one_failed_student_does_not_block_another(self):
        missing = RecognizedFace((0,0,1,1), 'Missing', 'missing')
        messages = self.session.check_in([missing, self.face], now=self.now)
        self.assertEqual(len(messages), 2)
        self.assertIn('failed', messages[0])
        self.assertIn('Checked in', messages[1])

    def test_stop_cancels_write_waiting_for_sqlite_lock(self):
        entered = Event()

        def waiting_write(*args, **kwargs):
            entered.set()
            return record_attendance(*args, **kwargs)

        with closing(sqlite3.connect(self.path)) as blocker:
            blocker.execute('BEGIN IMMEDIATE')
            with patch('core.attendance.record_attendance', side_effect=waiting_write), ThreadPoolExecutor(max_workers=1) as worker:
                future = worker.submit(self.session.check_in, [self.face], now=self.now)
                try:
                    self.assertTrue(entered.wait(2))
                    self.session.stop()
                finally:
                    blocker.rollback()
                self.assertEqual(future.result(timeout=3), [])
        self.assertEqual(get_attendance_history(self.path), [])
