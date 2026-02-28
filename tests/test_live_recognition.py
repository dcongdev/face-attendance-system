from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.live_recognition import RecognizedFace, label_for_box, recognize_frame
from database.db import initialize_database, insert_student
from gui.app import AttendanceApp


class LiveRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'students.sqlite3'
        initialize_database(self.path)
        self.embedding = np.ones(128, dtype=np.float32)
        self.frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detector = patch('core.live_recognition.FaceDetector')
        self.detector = detector.start().return_value
        self.addCleanup(detector.stop)
        self.detector.detect_faces.return_value = [(10, 20, 100, 100)]

    def recognize(self):
        return recognize_frame(self.frame, db_path=self.path)

    def test_known_and_unknown_multiple_faces(self):
        insert_student('S01', 'Nguyễn An', self.embedding, self.path)
        self.detector.detect_faces.return_value += [(200, 20, 100, 100)]
        with patch('core.live_recognition.extract_embedding', side_effect=[self.embedding, -self.embedding]):
            results, message = self.recognize()
        self.assertEqual([r.label for r in results], ['Nguyễn An (S01)', 'Unknown'])
        self.assertIn('running', message)

    def test_empty_database_skips_facenet(self):
        with patch('core.live_recognition.extract_embedding') as extract:
            results, message = self.recognize()
        self.assertEqual(results[0].label, 'Unknown')
        self.assertIn('No registered', message)
        extract.assert_not_called()

    def test_no_faces_skips_facenet(self):
        insert_student('S01', 'An', self.embedding, self.path)
        self.detector.detect_faces.return_value = []
        with patch('core.live_recognition.extract_embedding') as extract:
            self.assertEqual(self.recognize()[0], [])
        extract.assert_not_called()

    def test_bad_crop_does_not_prevent_other_faces(self):
        insert_student('S01', 'An', self.embedding, self.path)
        self.detector.detect_faces.return_value += [(200, 20, 100, 100)]
        with patch('core.live_recognition.extract_embedding', side_effect=[RuntimeError('Model error'), self.embedding]):
            results, message = self.recognize()
        self.assertEqual([r.label for r in results], ['Unknown', 'An (S01)'])
        self.assertIn('1 face', message)

    def test_registration_is_visible_on_next_pass(self):
        self.assertIn('No registered', self.recognize()[1])
        insert_student('S01', 'An', self.embedding, self.path)
        with patch('core.live_recognition.extract_embedding', return_value=self.embedding):
            self.assertEqual(self.recognize()[0][0].label, 'An (S01)')

    def test_labels_require_unambiguous_spatial_overlap(self):
        result = RecognizedFace((10, 20, 100, 100), 'An (S01)')
        self.assertEqual(label_for_box(result.box, [result]), result.label)
        self.assertEqual(label_for_box((200, 20, 100, 100), [result]), 'Unknown')
        self.assertEqual(label_for_box(result.box, [result, result]), 'Unknown')


class LiveRecognitionGuiTests(unittest.TestCase):
    def setUp(self):
        self.geometry = patch('gui.app.load_geometry', return_value='1000x650')
        self.geometry.start()
        self.addCleanup(self.geometry.stop)
        self.app = AttendanceApp()
        self.app.withdraw()
        self.addCleanup(self.cleanup)
        self.camera = Mock()
        self.camera.snapshot.return_value = np.zeros((160, 160, 3), dtype=np.uint8)
        self.camera.messages = Queue()
        self.app.camera = self.camera

    def cleanup(self):
        for callback in self.app.tk.call('after', 'info'):
            self.app.after_cancel(callback)
        self.app.destroy()

    def pump(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.app.update()
            time.sleep(.01)
        self.assertTrue(condition())

    def test_background_recognition_is_single_flight_and_throttled(self):
        release = Event()

        def recognize(frame):
            release.wait(3)
            return [], 'Live recognition running'

        with patch('gui.app.recognize_frame', side_effect=recognize) as recognize_mock:
            try:
                self.app._maybe_recognize()
                self.app._maybe_recognize()
                ticks = []
                self.app.after(10, lambda: ticks.append(True))
                self.pump(lambda: bool(ticks) and recognize_mock.called)
                self.assertEqual(recognize_mock.call_count, 1)
            finally:
                release.set()
                self.pump(lambda: not self.app._recognition_busy)
            self.app._maybe_recognize()
            self.assertEqual(recognize_mock.call_count, 1)
            self.camera.set_recognition.assert_called_once()

    def test_registration_and_extraction_pause_new_recognition(self):
        with patch('gui.app.recognize_frame') as recognize_mock:
            self.app._registration_busy = True
            self.app._maybe_recognize()
            self.app._registration_busy = False
            self.app._embedding_busy = True
            self.app._maybe_recognize()
            recognize_mock.assert_not_called()

    def test_stopped_or_replaced_camera_does_not_receive_old_results(self):
        self.app._recognition_busy = True
        self.app._recognition_results.put((self.camera, time.monotonic(), [], 'Old result'))
        self.app.camera = Mock()
        self.app._poll_recognition()
        self.camera.set_recognition.assert_not_called()
        self.app.camera.set_recognition.assert_not_called()

    def test_errors_are_reported_without_stopping_camera(self):
        with patch('gui.app.recognize_frame', side_effect=RuntimeError('Database locked')):
            self.app._maybe_recognize()
            self.pump(lambda: not self.app._recognition_busy)
        self.assertIn('Database locked', self.app._recognition_status)
        self.assertIs(self.app.camera, self.camera)
        self.camera.set_recognition.assert_called_once()
