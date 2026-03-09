from datetime import datetime, timezone
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import numpy as np

from database.db import (
    DatabaseError, get_student, initialize_database, insert_student, record_attendance,
    update_student_name,
)
from gui.app import AttendanceApp
from gui.attendance_history import AttendanceHistoryWindow
from gui.student_management import StudentManagementWindow


class StudentManagementGuiTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'students.sqlite3'
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.root.destroy()
        gc.collect()

    def pump(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertTrue(condition())

    def open(self, seed=True, on_updated=None):
        initialize_database(self.path)
        if seed:
            insert_student('S01', 'Nguyễn An', np.ones(128), self.path)
        self.window = StudentManagementWindow(self.root, self.path, on_updated)
        self.pump(lambda: not self.window._busy)
        return self.window

    def select(self):
        self.window.table.selection_set('S01')
        self.window._on_selection()

    def test_search_refresh_and_edit_with_immutable_id(self):
        changed = Mock()
        window = self.open(on_updated=changed)
        self.assertEqual(window.count_label.cget('text'), 'Students: 1')
        self.select()
        self.assertEqual(str(window.id_entry.cget('state')), 'readonly')
        window.full_name.set(' New Name ')
        window.save_button.invoke()
        self.pump(lambda: not window._busy)
        self.assertEqual(get_student('S01',self.path).full_name, 'New Name')
        changed.assert_called_once()
        self.assertIn('Saved name',window.status_label.cget('text'))
        window.query.set('missing')
        window.search_button.invoke()
        self.pump(lambda: not window._busy)
        self.assertEqual(window.table.get_children(), ())
        self.assertIn('No students',window.status_label.cget('text'))
        window.query.set('new name')
        window.refresh_button.invoke()
        self.pump(lambda: not window._busy)
        self.assertEqual(window.table.get_children(), ('S01',))

    def test_empty_list_selection_and_invalid_name(self):
        window = self.open(seed=False)
        self.assertIn('No students',window.status_label.cget('text'))
        window.save_name()
        self.assertIn('Select a student',window.status_label.cget('text'))
        insert_student('S01','An',np.ones(128),self.path)
        window.refresh()
        self.pump(lambda: not window._busy)
        self.select()
        window.full_name.set(' ')
        window.save_name()
        self.assertIn('Enter a full name',window.status_label.cget('text'))
        self.assertEqual(get_student('S01',self.path).full_name, 'An')

    def test_database_errors_restore_controls(self):
        window = self.open()
        self.select()
        with patch('gui.student_management.update_student_name', side_effect=DatabaseError('locked')):
            window.full_name.set('Other')
            window.save_name()
            self.pump(lambda: not window._busy)
        self.assertIn('locked',window.status_label.cget('text'))
        self.assertTrue(window.save_button.instate(['!disabled']))
        with patch('gui.student_management.search_students', side_effect=DatabaseError('load failed')):
            window.refresh()
            self.pump(lambda: not window._busy)
        self.assertIn('load failed',window.status_label.cget('text'))
        self.assertTrue(window.refresh_button.instate(['!disabled']))

    def test_save_runs_off_event_loop_and_rejects_double_submission(self):
        window = self.open()
        self.select()
        release = Event()

        def save(*args):
            release.wait(3)
            return update_student_name(*args)

        with patch('gui.student_management.update_student_name', side_effect=save) as writer:
            try:
                window.full_name.set('Other')
                window.save_name()
                window.save_name()
                ticks=[]
                self.root.after(10,lambda: ticks.append(True))
                self.pump(lambda: bool(ticks) and writer.called)
                self.assertEqual(writer.call_count,1)
            finally:
                release.set()
                self.pump(lambda: not window._busy)
        self.assertEqual(get_student('S01',self.path).full_name, 'Other')

    def test_profile_update_refreshes_history_and_invalidates_old_recognition(self):
        initialize_database(self.path)
        insert_student('S01','Old Name',np.ones(128),self.path)
        record_attendance('S01',self.path,now=datetime.now(timezone.utc))
        with patch('gui.app.load_geometry',return_value='1000x650'), patch(
            'gui.app.StudentManagementWindow',side_effect=lambda master, on_updated, capture_frame: StudentManagementWindow(master,self.path,on_updated,capture_frame)
        ) as factory:
            app=AttendanceApp()
            app.withdraw()
            try:
                app._history_window=AttendanceHistoryWindow(app,self.path)
                self.pump(lambda: not app._history_window._busy)
                app.camera=Mock()
                app._attendance_active=True
                previous_session=app._recognition_session
                app.open_student_management()
                app.open_student_management()
                self.assertEqual(factory.call_count,1)
                self.pump(lambda: not app._student_window._busy)
                manager=app._student_window
                manager.table.selection_set('S01')
                manager._on_selection()
                manager.full_name.set('New Name')
                manager.save_name()
                self.pump(lambda: not manager._busy and not app._history_window._busy)
                item=app._history_window.table.get_children()[0]
                self.assertEqual(app._history_window.table.item(item,'values')[1], 'New Name')
                self.assertGreater(app._recognition_session,previous_session)
                app.camera.enable_recognition.assert_called_with(True)
            finally:
                app.destroy()

    def test_update_face_uses_snapshot_and_not_unsaved_name(self):
        changed=Mock()
        window=self.open(on_updated=changed)
        self.select()
        frame=np.zeros((160,160,3),dtype=np.uint8)
        window.capture_frame=Mock(return_value=frame)
        window.full_name.set('Unsaved name')
        with patch('core.student_profile.extract_embedding',return_value=-np.ones(128,dtype=np.float32)):
            window.update_face_button.invoke()
            self.pump(lambda: not window._busy)
        window.capture_frame.assert_called_once()
        changed.assert_called_once()
        self.assertIn('Updated face',window.status_label.cget('text'))
        self.assertEqual(get_student('S01',self.path).full_name,'Nguyễn An')
        np.testing.assert_array_equal(get_student('S01',self.path).embedding,-np.ones(128,dtype=np.float32))

    def test_update_face_requires_selection_camera_and_one_face(self):
        window=self.open()
        window.update_face()
        self.assertIn('Select a student',window.status_label.cget('text'))
        self.select()
        window.update_face()
        self.assertIn('Start the camera',window.status_label.cget('text'))
        window.capture_frame=Mock(return_value=np.zeros((160,160,3),dtype=np.uint8))
        for error in (ValueError('No face detected'),ValueError('Multiple faces detected'),DatabaseError('locked')):
            with patch('gui.student_management.update_student_face',side_effect=error):
                window.update_face()
                self.pump(lambda: not window._busy)
            self.assertIn(str(error),window.status_label.cget('text'))
            self.assertTrue(window.update_face_button.instate(['!disabled']))
            np.testing.assert_array_equal(get_student('S01',self.path).embedding,np.ones(128))

    def test_face_processing_is_background_and_submissions_are_single(self):
        window=self.open()
        self.select()
        window.capture_frame=Mock(return_value=np.zeros((160,160,3),dtype=np.uint8))
        release=Event()

        def process(*args):
            release.wait(3)
            raise ValueError('No face detected')

        with patch('gui.student_management.update_student_face',side_effect=process) as update:
            try:
                window.update_face()
                window.update_face()
                self.assertTrue(window.update_face_button.instate(['disabled']))
                ticks=[]
                self.root.after(10,lambda: ticks.append(True))
                self.pump(lambda: bool(ticks) and update.called)
                self.assertEqual(update.call_count,1)
            finally:
                release.set()
                self.pump(lambda: not window._busy)

    def test_main_camera_snapshot_and_face_update_invalidate_old_recognition(self):
        initialize_database(self.path)
        insert_student('S01','An',np.ones(128),self.path)
        with patch('gui.app.load_geometry',return_value='1000x650'), patch(
            'gui.app.StudentManagementWindow',side_effect=lambda master, on_updated, capture_frame:
                StudentManagementWindow(master,self.path,on_updated,capture_frame)
        ):
            app=AttendanceApp()
            app.withdraw()
            try:
                with self.assertRaises(ValueError):
                    app._capture_student_face()
                app.camera=Mock()
                frame=np.zeros((160,160,3),dtype=np.uint8)
                app.camera.snapshot.return_value=frame
                app._attendance_active=True
                generation=app._recognition_session
                app.open_student_management()
                manager=app._student_window
                self.pump(lambda: not manager._busy)
                manager.table.selection_set('S01')
                manager._on_selection()
                with patch('core.student_profile.extract_embedding',return_value=-np.ones(128,dtype=np.float32)):
                    manager.update_face_button.invoke()
                    self.pump(lambda: not manager._busy)
                app.camera.snapshot.assert_called_once()
                self.assertGreater(app._recognition_session,generation)
                app.camera.enable_recognition.assert_called_with(True)
                self.assertIn('Updated face',manager.status_label.cget('text'))
            finally:
                app.destroy()

    def test_delete_student_confirmation_cancel_and_history_preservation(self):
        changed=Mock()
        window=self.open(on_updated=changed)
        record_attendance('S01',self.path,now=datetime.now(timezone.utc))
        window.delete_selected_student()
        self.assertIn('Select a student',window.status_label.cget('text'))
        self.select()
        window.full_name.set('Unsaved name')
        with patch('gui.student_management.messagebox.askyesno',return_value=False) as confirm:
            window.delete_selected_student()
        self.assertIn('S01',confirm.call_args.args[1])
        self.assertIn('Nguyễn An',confirm.call_args.args[1])
        self.assertNotIn('Unsaved name',confirm.call_args.args[1])
        self.assertIsNotNone(get_student('S01',self.path))
        changed.assert_not_called()
        with patch('gui.student_management.messagebox.askyesno',return_value=True):
            window.delete_button.invoke()
            self.pump(lambda: not window._busy)
        changed.assert_called_once()
        self.assertIsNone(get_student('S01',self.path))
        self.assertEqual(window.count_label.cget('text'),'Students: 0')
        self.assertIn('attendance history preserved',window.status_label.cget('text'))
        from database.db import get_attendance_history
        self.assertEqual(get_attendance_history(self.path)[0].full_name,'Nguyễn An')

    def test_student_deletion_errors_restore_controls(self):
        window=self.open()
        self.select()
        with patch('gui.student_management.messagebox.askyesno',return_value=True), patch(
            'gui.student_management.delete_student',side_effect=DatabaseError('locked')
        ):
            window.delete_selected_student()
            self.pump(lambda: not window._busy)
        self.assertIn('locked',window.status_label.cget('text'))
        self.assertTrue(window.delete_button.instate(['!disabled']))
        self.assertIsNotNone(get_student('S01',self.path))

    def test_student_deletion_invalidates_recognition_without_changing_modes(self):
        initialize_database(self.path)
        insert_student('S01','An',np.ones(128),self.path)
        with patch('gui.app.load_geometry',return_value='1000x650'), patch(
            'gui.app.StudentManagementWindow',side_effect=lambda master,on_updated,capture_frame:
                StudentManagementWindow(master,self.path,on_updated,capture_frame)
        ), patch('gui.student_management.messagebox.askyesno',return_value=True):
            app=AttendanceApp()
            app.withdraw()
            try:
                app.camera=Mock()
                app._attendance_active=True
                session=app._attendance_session
                generation=app._recognition_session
                app.open_student_management()
                manager=app._student_window
                self.pump(lambda: not manager._busy)
                manager.table.selection_set('S01')
                manager._on_selection()
                manager.delete_button.invoke()
                self.pump(lambda: not manager._busy)
                self.assertGreater(app._recognition_session,generation)
                self.assertTrue(app._attendance_active)
                self.assertIs(app._attendance_session,session)
                app.camera.stop.assert_not_called()
                app.camera.enable_recognition.assert_called_with(True)
                self.assertEqual(manager.table.get_children(),())
            finally:
                app.destroy()
