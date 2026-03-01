from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

from database.db import (
    ATTENDANCE_TIMEZONE, DatabaseError, initialize_database, insert_student, record_attendance,
)
from gui.app import AttendanceApp
from gui.attendance_history import AttendanceHistoryWindow


class AttendanceHistoryGuiTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'attendance.sqlite3'
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.window = None
        self.now = datetime.now(ATTENDANCE_TIMEZONE)

    def open_window(self):
        self.window = AttendanceHistoryWindow(self.root, self.path)
        self.pump(lambda: not self.window._busy)
        return self.window

    def pump(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertTrue(condition())

    def rows(self):
        return [self.window.table.item(item, 'values') for item in self.window.table.get_children()]

    def seed(self):
        initialize_database(self.path)
        insert_student('S01', 'Nguyễn An', np.ones(128), self.path)
        record_attendance('S01', self.path, now=self.now - timedelta(days=1))
        record_attendance('S01', self.path, now=self.now)

    def test_today_default_and_all_records_with_local_time(self):
        self.seed()
        window = self.open_window()
        self.assertFalse(window.show_all.get())
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0], ('S01', 'Nguyễn An', self.now.date().isoformat(), self.now.strftime('%H:%M:%S')))
        window.filter_button.invoke()
        self.pump(lambda: not window._busy)
        self.assertEqual(len(self.rows()), 2)
        window.filter_button.invoke()
        self.pump(lambda: not window._busy)
        self.assertEqual(len(self.rows()), 1)

    def test_refresh_loads_new_records(self):
        self.seed()
        window = self.open_window()
        insert_student('S02', 'Binh', np.ones(128), self.path)
        record_attendance('S02', self.path, now=self.now)
        window.refresh_button.invoke()
        self.pump(lambda: not window._busy)
        self.assertEqual(len(self.rows()), 2)

    def test_fresh_database_and_empty_today(self):
        window = self.open_window()
        self.assertEqual(self.rows(), [])
        self.assertIn('No attendance records', window.status_label.cget('text'))
        insert_student('S01', 'An', np.ones(128), self.path)
        record_attendance('S01', self.path, now=self.now - timedelta(days=1))
        window.refresh()
        self.pump(lambda: not window._busy)
        self.assertEqual(self.rows(), [])

    def test_database_error_clears_stale_rows_and_can_retry(self):
        self.seed()
        window = self.open_window()
        with patch('gui.attendance_history.get_attendance_history', side_effect=DatabaseError('locked')):
            window.refresh()
            self.pump(lambda: not window._busy)
        self.assertEqual(self.rows(), [])
        self.assertIn('locked', window.status_label.cget('text'))
        self.assertTrue(window.refresh_button.instate(['!disabled']))
        window.refresh()
        self.pump(lambda: not window._busy)
        self.assertEqual(len(self.rows()), 1)

    def test_background_loading_is_responsive_and_close_is_safe(self):
        release, completed = Event(), Event()

        def load(path):
            release.wait(3)
            completed.set()
            return []

        with patch('gui.attendance_history.get_attendance_history', side_effect=load) as load_mock:
            window = AttendanceHistoryWindow(self.root, self.path)
            try:
                ticks = []
                self.root.after(10, lambda: ticks.append(True))
                self.pump(lambda: bool(ticks) and load_mock.called)
                self.assertTrue(window._busy)
                window.refresh()
                self.assertEqual(load_mock.call_count, 1)
                window.destroy()
                self.assertIsNone(window._poll_id)
            finally:
                release.set()
                self.assertTrue(completed.wait(3))
            self.root.update()

    def test_main_gui_reuses_window_without_changing_workflows(self):
        with patch('gui.app.load_geometry', return_value='1000x650'), patch(
            'gui.app.AttendanceHistoryWindow', side_effect=lambda master: AttendanceHistoryWindow(master, self.path)
        ) as factory:
            app = AttendanceApp()
            app.withdraw()
            try:
                app.open_attendance_history()
                window = app._history_window
                app.open_attendance_history()
                self.assertEqual(factory.call_count, 1)
                self.assertIsNone(app.camera)
                self.assertFalse(app._attendance_active)
                self.assertFalse(app._registration_busy)
                window.destroy()
                app.open_attendance_history()
                self.assertEqual(factory.call_count, 2)
                self.pump(lambda: not app._history_window._busy)
            finally:
                app.destroy()
