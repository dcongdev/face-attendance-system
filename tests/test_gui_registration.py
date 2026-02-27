from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, get_ident
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.registration import RegistrationResult, register_student
from database.db import get_student
from gui.app import AttendanceApp


class GuiRegistrationTests(unittest.TestCase):
    def setUp(self):
        geometry = patch('gui.app.load_geometry', return_value='1000x650')
        geometry.start()
        self.addCleanup(geometry.stop)
        self.app = AttendanceApp()
        self.app.withdraw()
        self.addCleanup(self.destroy_app)
        self.frame = np.zeros((160,160,3), dtype=np.uint8)
        self.app.camera = Mock()
        self.app.camera.snapshot.return_value = self.frame.copy()
        self.app.camera.messages = Queue()
        self.app.student_id_entry.insert(0, ' S01 ')
        self.app.student_name_entry.insert(0, ' An ')

    def destroy_app(self):
        if self.app_exists():
            for callback in self.app.tk.call('after', 'info'):
                self.app.after_cancel(callback)
            self.app.destroy()

    def app_exists(self):
        try:
            return bool(self.app.winfo_exists())
        except tk.TclError:
            return False

    def pump_until(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.app.update()
            time.sleep(.01)
        self.assertTrue(condition())

    def test_success_persists_and_duplicate_is_reported(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'students.sqlite3'
            embedding = np.ones(128, dtype=np.float32)
            with patch('core.registration.extract_embedding', return_value=embedding), patch(
                'gui.app.register_student', side_effect=lambda *args: register_student(*args, db_path=path)
            ):
                self.app.register_button.invoke()
                self.pump_until(lambda: not self.app._registration_busy)
                self.assertIn('successfully', self.app.status_label.cget('text'))
                self.assertEqual(get_student('S01', path).full_name, 'An')
                self.app.register_button.invoke()
                self.pump_until(lambda: not self.app._registration_busy)
                self.assertIn('already registered', self.app.status_label.cget('text'))

    def test_background_work_blocks_duplicate_submission_but_not_tk(self):
        release = Event()
        main_thread = get_ident()
        worker_threads = []

        def backend(*args):
            worker_threads.append(get_ident())
            release.wait(3)
            return RegistrationResult(True, 'success', 'Saved S01')

        with patch('gui.app.register_student', side_effect=backend) as backend_mock, patch(
            'gui.app.extract_embedding'
        ) as extract:
            try:
                self.app.register_student()
                self.app.register_student()
                self.app.extract_embedding()
                self.assertTrue(self.app.register_button.instate(['disabled']))
                self.assertTrue(self.app.extract_button.instate(['disabled']))
                ticks = []
                self.app.after(10, lambda: ticks.append(True))
                self.pump_until(lambda: bool(ticks) and bool(worker_threads))
                self.assertNotEqual(worker_threads[0], main_thread)
                self.assertEqual(backend_mock.call_count, 1)
                extract.assert_not_called()
                args = backend_mock.call_args.args
                self.assertEqual(args[:2], ('S01','An'))
                np.testing.assert_array_equal(args[2], self.frame)
            finally:
                release.set()
                self.pump_until(lambda: not self.app._registration_busy)
        self.assertTrue(self.app.register_button.instate(['!disabled']))
        self.assertTrue(self.app.extract_button.instate(['!disabled']))

    def test_missing_inputs_and_inactive_camera(self):
        with patch('gui.app.register_student') as backend:
            self.app.student_id_entry.delete(0, 'end')
            self.app.register_student()
            self.assertIn('student ID', self.app.status_label.cget('text'))
            self.app.student_id_entry.insert(0,'S01')
            self.app.student_name_entry.delete(0,'end')
            self.app.register_student()
            self.assertIn('full name', self.app.status_label.cget('text'))
            self.app.student_name_entry.insert(0,'An')
            self.app.camera = None
            self.app.register_student()
            self.assertIn('No camera frame', self.app.status_label.cget('text'))
            backend.assert_not_called()

    def test_errors_restore_controls(self):
        for response in (RegistrationResult(False,'extraction_error','Multiple faces detected'),
                         RegistrationResult(False,'database_error','Database locked'),
                         RuntimeError('Unexpected failure')):
            with self.subTest(response=response):
                kwargs = {'side_effect':response} if isinstance(response,Exception) else {'return_value':response}
                with patch('gui.app.register_student', **kwargs):
                    self.app.register_student()
                    self.pump_until(lambda: not self.app._registration_busy)
                    message = str(response) if isinstance(response,Exception) else response.message
                    self.assertIn(message, self.app.status_label.cget('text'))
                    self.assertTrue(self.app.register_button.instate(['!disabled']))

    def test_result_survives_preview_and_embedding_remains_available(self):
        with patch('gui.app.register_student', return_value=RegistrationResult(True,'success','Saved S01')):
            self.app.register_student()
            self.pump_until(lambda: not self.app._registration_busy)
        self.app.camera.messages.put(('frame', self.frame))
        self.app._update_preview()
        self.assertIn('Saved S01', self.app.status_label.cget('text'))
        with patch('gui.app.extract_embedding', return_value=np.ones(128)):
            self.app.extract_embedding()
            self.pump_until(lambda: not self.app._embedding_busy)
        self.assertIn('128 dimensions', self.app.status_label.cget('text'))

    def test_close_waits_for_registration_and_camera_cleanup(self):
        release = Event()

        def backend(*args):
            release.wait(3)
            return RegistrationResult(True,'success','Saved')

        with patch('gui.app.register_student', side_effect=backend), patch('gui.app.save_geometry'):
            try:
                self.app.register_student()
                self.app.close()
                self.app.camera.stop.assert_called_once()
                self.app.camera.messages.put(('stopped', None))
                self.app._update_preview()
                self.assertTrue(self.app.winfo_exists())
            finally:
                release.set()
            self.pump_until(lambda: not self.app_exists())


if __name__ == '__main__':
    unittest.main()
