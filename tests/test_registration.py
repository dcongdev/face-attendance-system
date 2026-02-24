from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.registration import register_student
from database.db import (
    DatabaseError, get_student, initialize_database, insert_student, load_registered_students,
)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data" / "students.sqlite3"
        self.frame = np.zeros((160, 160, 3), dtype=np.uint8)
        self.embedding = np.linspace(-1, 1, 128, dtype=np.float32)
        self.extractor = patch('core.registration.extract_embedding', return_value=self.embedding).start()
        self.addCleanup(patch.stopall)

    def register(self, student_id="S01", full_name="An", frame=None):
        return register_student(student_id, full_name, self.frame if frame is None else frame, self.path)

    def test_success_creates_storage_and_restores_embedding(self):
        result = self.register(" S01 ", " Nguyễn Văn An ")
        self.assertTrue(result.success)
        self.assertEqual(result.code, "success")
        self.assertEqual(result.student.student_id, "S01")
        stored = get_student("S01", self.path)
        self.assertEqual(stored.full_name, "Nguyễn Văn An")
        self.assertEqual(stored.registered_at, result.student.registered_at)
        np.testing.assert_array_equal(stored.embedding, self.embedding)
        self.extractor.assert_called_once_with(self.frame)

    def test_required_fields_fail_before_database_or_extraction(self):
        for student_id, name in [(None,"An"), (" ","An"), (123,"An"),
                                 ("Unknown","An"), ("S01",None), ("S01"," ")]:
            with self.subTest(student_id=student_id, name=name):
                result = self.register(student_id, name)
                self.assertEqual(result.code, "invalid_input")
                self.assertFalse(result.success)
                self.assertIsNone(result.student)
        self.extractor.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_invalid_frames(self):
        for frame in (None, [], np.zeros((0,0,3),dtype=np.uint8),
                      np.zeros((10,10),dtype=np.uint8), np.zeros((10,10,4),dtype=np.uint8),
                      np.zeros((10,10,3),dtype=float)):
            with self.subTest(shape=getattr(frame,"shape",None)):
                result = register_student("S01", "An", frame, self.path)
                self.assertEqual(result.code, "invalid_input")
        self.extractor.assert_not_called()

    def test_duplicate_rejected_before_inference(self):
        self.assertTrue(self.register().success)
        self.extractor.reset_mock()
        result = self.register(" S01 ", "Other")
        self.assertEqual(result.code, "duplicate_student")
        self.extractor.assert_not_called()
        self.assertEqual(get_student("S01", self.path).full_name, "An")

    def test_face_errors_and_runtime_failures_do_not_insert(self):
        for error in (ValueError("No face detected."), ValueError("Multiple faces detected."),
                      RuntimeError("Model unavailable")):
            with self.subTest(error=error):
                self.extractor.side_effect = error
                result = self.register()
                self.assertFalse(result.success)
                self.assertEqual(result.code, "extraction_error")
                self.assertIn(str(error), result.message)
                self.assertEqual(load_registered_students(self.path), {})

    def test_invalid_extracted_embedding_is_rejected(self):
        for embedding in ([1,2], np.zeros(128), [np.nan]*128):
            with self.subTest(embedding=str(embedding)[:30]):
                self.extractor.return_value = embedding
                self.assertEqual(self.register().code, "extraction_error")
                self.assertEqual(load_registered_students(self.path), {})

    def test_database_initialization_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("not sqlite")
        self.assertEqual(self.register().code, "database_error")
        self.extractor.assert_not_called()

    def test_database_insert_error(self):
        with patch('core.registration.insert_student', side_effect=DatabaseError("Database locked")):
            result = self.register()
        self.assertEqual(result.code, "database_error")
        self.assertIn("locked", result.message)
        self.assertEqual(load_registered_students(self.path), {})

    def test_duplicate_insert_race_does_not_overwrite(self):
        initialize_database(self.path)

        def racing_extractor(frame):
            insert_student("S01", "Other writer", self.embedding, self.path)
            return self.embedding

        self.extractor.side_effect = racing_extractor
        self.assertEqual(self.register().code, "duplicate_student")
        self.assertEqual(get_student("S01", self.path).full_name, "Other writer")

    def test_real_extraction_helper_rejects_face_counts_before_represent(self):
        from core.face_embedding import extract_embedding

        self.extractor.side_effect = extract_embedding
        api = Mock()
        with patch.dict('sys.modules', {'deepface': types.SimpleNamespace(DeepFace=api)}):
            for faces in ([], [{'face': self.frame}]*2):
                api.extract_faces.return_value = faces
                result = self.register()
                self.assertEqual(result.code, "extraction_error")
                self.assertEqual(load_registered_students(self.path), {})
        api.represent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
