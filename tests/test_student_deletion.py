from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from core.live_recognition import recognize_frame
from database.db import (
    DatabaseError, StudentNotFoundError, delete_student, get_attendance_history, get_student,
    initialize_database, insert_student, load_registered_students, record_attendance,
    search_students, update_student_name,
)


class StudentDeletionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'students.sqlite3'
        initialize_database(self.path)
        self.embedding = np.ones(128, dtype=np.float32)
        insert_student('S01','Nguyễn An',self.embedding,self.path)
        self.now = datetime(2026,10,16,3,tzinfo=timezone.utc)
        record_attendance('S01',self.path,now=self.now)

    def test_deletes_profile_and_embedding_but_preserves_complete_history(self):
        insert_student('S02','Binh',-self.embedding,self.path)
        history = get_attendance_history(self.path)
        deleted = delete_student(' S01 ',self.path)
        self.assertEqual(deleted.full_name,'Nguyễn An')
        self.assertIsNone(get_student('S01',self.path))
        self.assertNotIn('S01',load_registered_students(self.path))
        self.assertEqual(search_students('S01',self.path),[])
        self.assertEqual(get_attendance_history(self.path),history)
        self.assertIsNotNone(get_student('S02',self.path))
        initialize_database(self.path)
        self.assertEqual(get_attendance_history(self.path),history)

    def test_freezes_latest_name_and_reregistration_cannot_change_old_history(self):
        update_student_name('S01','Updated Name',self.path)
        delete_student('S01',self.path)
        self.assertEqual(get_attendance_history(self.path)[0].full_name,'Updated Name')
        insert_student('S01','New Profile',self.embedding,self.path)
        record_attendance('S01',self.path,now=self.now+timedelta(days=1))
        update_student_name('S01','New Profile Renamed',self.path)
        self.assertEqual([r.full_name for r in get_attendance_history(self.path)],
                         ['New Profile Renamed','Updated Name'])

    def test_student_without_attendance_can_be_deleted(self):
        insert_student('S02','Binh',self.embedding,self.path)
        history=get_attendance_history(self.path)
        delete_student('S02',self.path)
        self.assertEqual(get_attendance_history(self.path),history)

    def test_missing_and_invalid_ids_do_not_change_data(self):
        with self.assertRaises(StudentNotFoundError):
            delete_student('missing',self.path)
        for student_id in (None,'',' ','Unknown',123):
            with self.subTest(student_id=student_id),self.assertRaises(ValueError):
                delete_student(student_id,self.path)
        self.assertIsNotNone(get_student('S01',self.path))
        self.assertEqual(len(get_attendance_history(self.path)),1)

    def test_database_failure_rolls_back_profile_and_history_snapshot(self):
        update_student_name('S01','Updated Name',self.path)
        history=get_attendance_history(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('''CREATE TRIGGER block_student_delete AFTER DELETE ON students
                                  BEGIN SELECT RAISE(ABORT,'blocked'); END''')
        with self.assertRaises(DatabaseError):
            delete_student('S01',self.path)
        self.assertIsNotNone(get_student('S01',self.path))
        self.assertEqual(get_attendance_history(self.path),history)
        with closing(sqlite3.connect(self.path)) as connection:
            name,ref=connection.execute('SELECT full_name, student_ref FROM attendance').fetchone()
        self.assertEqual((name,ref),('Nguyễn An','S01'))
        with self.assertRaises(DatabaseError):
            delete_student('S01',self.path.parent/'missing.sqlite3')

    def test_deleted_profile_is_unknown_on_next_recognition(self):
        frame=np.zeros((160,160,3),dtype=np.uint8)
        with patch('core.live_recognition.FaceDetector') as detector, patch(
            'core.live_recognition.extract_embedding',return_value=self.embedding
        ):
            detector.return_value.detect_faces.return_value=[(0,0,160,160)]
            self.assertEqual(recognize_frame(frame,db_path=self.path)[0][0].student_id,'S01')
            delete_student('S01',self.path)
            result=recognize_frame(frame,db_path=self.path)[0][0]
            self.assertIsNone(result.student_id)
            self.assertEqual(result.label,'Unknown')

    def test_previous_schema_migrates_preserving_ids_and_sequence(self):
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('DROP TABLE attendance')
                connection.execute('''CREATE TABLE attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    attendance_date TEXT NOT NULL, checked_in_at TEXT NOT NULL,
                    UNIQUE(student_id,attendance_date))''')
                connection.execute('INSERT INTO attendance VALUES (31,?,?,?)',
                                   ('S01','2026-10-16',self.now.isoformat()))
                connection.execute("UPDATE sqlite_sequence SET seq=52 WHERE name='attendance'")
        initialize_database(self.path)
        self.assertEqual(get_attendance_history(self.path)[0].record_id,31)
        delete_student('S01',self.path)
        self.assertEqual(get_attendance_history(self.path)[0].full_name,'Nguyễn An')
        insert_student('S02','Binh',self.embedding,self.path)
        record_attendance('S02',self.path,now=self.now)
        self.assertGreater(max(r.record_id for r in get_attendance_history(self.path)),52)

    def test_migration_failure_keeps_previous_schema_and_records(self):
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('DROP TABLE attendance')
                connection.execute('''CREATE TABLE attendance (
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    attendance_date TEXT NOT NULL, checked_in_at TEXT NOT NULL,
                    PRIMARY KEY(student_id,attendance_date))''')
                connection.execute('INSERT INTO attendance VALUES (?,?,?)',
                                   ('S01','2026-10-16',self.now.isoformat()))
                connection.execute('CREATE TABLE attendance_migration (placeholder TEXT)')
        with self.assertRaises(DatabaseError):
            initialize_database(self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute('SELECT * FROM attendance').fetchall(),
                             [('S01','2026-10-16',self.now.isoformat())])
            self.assertNotIn('full_name',[r[1] for r in connection.execute('PRAGMA table_info(attendance)')])
