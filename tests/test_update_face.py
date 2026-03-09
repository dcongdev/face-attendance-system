from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.live_recognition import recognize_frame
from core.student_profile import update_student_face
from database.db import (
    DatabaseError, StudentNotFoundError, get_attendance_history, get_student,
    initialize_database, insert_student, record_attendance, update_student_embedding,
)


class UpdateFaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'students.sqlite3'
        initialize_database(self.path)
        self.old_embedding = np.ones(128, dtype=np.float32)
        self.new_embedding = -self.old_embedding
        insert_student('S01','An',self.old_embedding,self.path)
        record_attendance('S01',self.path,now=datetime(2026,10,16,tzinfo=timezone.utc))
        self.frame = np.zeros((160,160,3),dtype=np.uint8)

    def assert_old_embedding(self):
        np.testing.assert_array_equal(get_student('S01',self.path).embedding,self.old_embedding)

    def test_success_changes_only_embedding(self):
        before=get_student('S01',self.path)
        attendance=get_attendance_history(self.path)
        with patch('core.student_profile.extract_embedding',return_value=self.new_embedding) as extract:
            updated=update_student_face('S01',self.frame,self.path)
        extract.assert_called_once_with(self.frame)
        after=get_student('S01',self.path)
        self.assertEqual((after.student_id,after.full_name,after.registered_at),
                         (before.student_id,before.full_name,before.registered_at))
        self.assertEqual(updated.student_id,'S01')
        np.testing.assert_array_equal(after.embedding,self.new_embedding)
        self.assertEqual(get_attendance_history(self.path),attendance)

    def test_invalid_frames_and_missing_students_skip_extraction(self):
        with patch('core.student_profile.extract_embedding') as extract:
            for frame in (None,[],np.zeros((0,0,3),dtype=np.uint8),np.zeros((5,5),dtype=np.uint8),
                          np.zeros((5,5,3),dtype=float)):
                with self.subTest(frame=str(type(frame))),self.assertRaises(ValueError):
                    update_student_face('S01',frame,self.path)
            with self.assertRaises(StudentNotFoundError):
                update_student_face('missing',self.frame,self.path)
            extract.assert_not_called()
        self.assert_old_embedding()

    def test_zero_multiple_faces_and_extraction_failures_keep_old_embedding(self):
        api=Mock()
        with patch.dict(sys.modules,{'deepface':types.SimpleNamespace(DeepFace=api)}):
            for faces in ([],[{'face':self.frame}]*2):
                api.extract_faces.return_value=faces
                with self.assertRaises(ValueError):
                    update_student_face('S01',self.frame,self.path)
                self.assert_old_embedding()
            api.represent.assert_not_called()
        with patch('core.student_profile.extract_embedding',side_effect=RuntimeError('Model unavailable')):
            with self.assertRaisesRegex(RuntimeError,'Model unavailable'):
                update_student_face('S01',self.frame,self.path)
        self.assert_old_embedding()

    def test_invalid_embeddings_keep_old_vector(self):
        for embedding in ([1,2],np.zeros(128),[np.nan]*128,[np.inf]*128):
            with self.subTest(embedding=str(embedding)[:20]),self.assertRaises(ValueError):
                update_student_embedding('S01',embedding,self.path)
            self.assert_old_embedding()

    def test_database_failure_rolls_back_embedding_update(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('''CREATE TRIGGER reject_face AFTER UPDATE OF embedding ON students
                                  BEGIN SELECT RAISE(ABORT,'blocked'); END''')
        with patch('core.student_profile.extract_embedding',return_value=self.new_embedding):
            with self.assertRaises(DatabaseError):
                update_student_face('S01',self.frame,self.path)
        self.assert_old_embedding()

    def test_missing_student_is_never_created(self):
        with self.assertRaises(StudentNotFoundError):
            update_student_embedding('missing',self.new_embedding,self.path)
        self.assertIsNone(get_student('missing',self.path))
        self.assert_old_embedding()

    def test_next_recognition_uses_new_embedding_without_restart(self):
        with patch('core.live_recognition.FaceDetector') as detector, patch(
            'core.live_recognition.extract_embedding',return_value=self.new_embedding
        ):
            detector.return_value.detect_faces.return_value=[(0,0,160,160)]
            self.assertIsNone(recognize_frame(self.frame,db_path=self.path)[0][0].student_id)
            with patch('core.student_profile.extract_embedding',return_value=self.new_embedding):
                update_student_face('S01',self.frame,self.path)
            self.assertEqual(recognize_frame(self.frame,db_path=self.path)[0][0].student_id,'S01')
