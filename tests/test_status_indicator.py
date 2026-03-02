import time
import tkinter as tk
import unittest

from gui.status_indicator import StatusIndicator


class StatusIndicatorTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.status = StatusIndicator(self.root)

    def pump(self, condition):
        deadline = time.monotonic() + 3
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertTrue(condition())

    def test_semantic_colors_and_labels(self):
        for state in ('idle', 'unknown', 'recognized', 'checked_in', 'already_checked_in'):
            with self.subTest(state=state):
                self.status.set(state, state)
                self.assertEqual(self.status.light.itemcget(self.status._dot, 'fill'), self.status.COLORS[state])
                self.assertIn(state, self.status.label.cget('text'))
        self.assertEqual(self.status.COLORS['recognized'], self.status.COLORS['checked_in'])
        self.assertNotEqual(self.status.COLORS['checked_in'], self.status.COLORS['already_checked_in'])

    def test_registration_blinks_in_event_loop_and_stops_on_completion(self):
        self.status.set('registering', 'Registering...')
        first_callback = self.status._blink_id
        self.pump(lambda: not self.status._lit)
        self.assertEqual(self.status.light.itemcget(self.status._dot, 'fill'), self.status.COLORS['idle'])
        callback = self.status._blink_id
        self.status.set('registered', 'Registered successfully')
        self.assertIsNone(self.status._blink_id)
        pending = self.root.tk.call('after', 'info')
        self.assertNotIn(first_callback, pending)
        self.assertNotIn(callback, pending)

    def test_checkin_hold_defers_latest_recognition_without_extending_timer(self):
        self.status.set('checked_in', 'Checked in: S01', hold_ms=80)
        callback = self.status._hold_id
        self.status.set('recognized', 'Recognized: S01')
        self.status.set('unknown', 'Unknown face')
        self.assertEqual(self.status._hold_id, callback)
        self.assertEqual(self.status.state, 'checked_in')
        self.assertIn('Checked in:', self.status.label.cget('text'))
        self.pump(lambda: self.status._hold_id is None)
        self.assertEqual(self.status.state, 'unknown')
        self.assertIn('Unknown', self.status.label.cget('text'))

    def test_force_stop_clears_hold_and_pending_recognition(self):
        self.status.set('already_checked_in', 'Already checked in', hold_ms=3000)
        callback = self.status._hold_id
        self.status.set('unknown', 'Unknown')
        self.status.set('idle', 'Stopped', force=True)
        self.assertIsNone(self.status._hold_id)
        self.assertIsNone(self.status._pending)
        self.assertNotIn(callback, self.root.tk.call('after', 'info'))

    def test_destroy_cancels_animation_and_hold(self):
        self.status.set('registering', 'Registering', hold_ms=3000)
        callbacks = (self.status._blink_id, self.status._hold_id)
        self.status.destroy()
        for callback in callbacks:
            self.assertNotIn(callback, self.root.tk.call('after', 'info'))
