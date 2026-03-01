from datetime import datetime, timezone
from threading import Event

from database.db import ATTENDANCE_TIMEZONE, DEFAULT_DB_PATH, record_attendance


class AttendanceSession:
    """Log structured recognition results in a worker; Stop cancels pending writes."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self.cancelled = Event()
        self._checked = set()
        self._errors = {}

    def stop(self):
        self.cancelled.set()

    def check_in(self, faces, *, now=None):
        """Return only new feedback, once per student/day or distinct error.

        Each session has one worker at a time. SQLite remains authoritative for
        duplicates across sessions/restarts. Unknown labels are never parsed.
        """
        now = now or datetime.now(timezone.utc)
        day = now.astimezone(ATTENDANCE_TIMEZONE).date().isoformat()
        messages = []
        for face in faces:
            if self.cancelled.is_set():
                break
            student_id = face.student_id
            if not student_id or student_id == "Unknown":
                continue
            key = (day, student_id)
            if key in self._checked:
                continue
            try:
                inserted = record_attendance(student_id, self.db_path, now=now,
                                             cancel_event=self.cancelled)
                if inserted is None:
                    break
            except Exception as exc:
                message = f"Check-in failed for {student_id}: {exc}"
                if self._errors.get(key) != message:
                    messages.append(message)
                    self._errors[key] = message
                continue
            self._checked.add(key)
            self._errors.pop(key, None)
            if inserted:
                messages.append(f"Checked in: {face.label}")
            else:
                messages.append(f"Already checked in today: {face.label}")
        return messages
