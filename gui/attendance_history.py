from datetime import datetime
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk

from database.db import (
    ATTENDANCE_TIMEZONE, DEFAULT_DB_PATH, get_attendance_history, initialize_database,
)


class AttendanceHistoryWindow(tk.Toplevel):
    """Read-only attendance table with background refresh and a local-day filter."""

    def __init__(self, master, db_path=DEFAULT_DB_PATH):
        super().__init__(master)
        self.title("Attendance History")
        self.geometry("820x450")
        self.minsize(650, 300)
        self.db_path = db_path
        self._results = Queue()
        self._busy = False
        self._closed = False
        self._poll_id = None
        self.show_all = tk.BooleanVar(self, value=False)

        content = ttk.Frame(self, padding=15)
        content.pack(fill="both", expand=True)
        controls = ttk.Frame(content)
        controls.pack(fill="x", pady=(0, 10))
        self.filter_button = ttk.Checkbutton(
            controls, text="Show all records", variable=self.show_all, command=self.refresh,
        )
        self.filter_button.pack(side="left")
        self.refresh_button = ttk.Button(controls, text="Refresh", command=self.refresh)
        self.refresh_button.pack(side="right")

        table_frame = ttk.Frame(content)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(
            table_frame, columns=("student_id", "full_name", "date", "time"),
            show="headings", selectmode="browse",
        )
        for column, title, width in (
            ("student_id", "Student ID", 120), ("full_name", "Full Name", 260),
            ("date", "Date", 130), ("time", "Check-in Time", 150),
        ):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, minwidth=90)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.status_label = ttk.Label(content, text="", wraplength=600)
        self.status_label.pack(fill="x", pady=(10, 0))
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()

    def refresh(self):
        if self._closed or self._busy:
            return
        self._busy = True
        self.refresh_button.configure(state="disabled")
        self.filter_button.configure(state="disabled")
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        self.status_label.configure(text="Loading attendance...")
        today = datetime.now(ATTENDANCE_TIMEZONE).date().isoformat()
        Thread(target=self._load, args=(self.show_all.get(), today), daemon=True).start()
        self._poll_id = self.after(50, self._poll)

    def _load(self, show_all, today):
        try:
            initialize_database(self.db_path)
            records = get_attendance_history(self.db_path)
            rows = [
                (record.student_id, record.full_name, record.attendance_date,
                 datetime.fromisoformat(record.checked_in_at).astimezone(ATTENDANCE_TIMEZONE).strftime("%H:%M:%S"))
                for record in records if show_all or record.attendance_date == today
            ]
            self._results.put((rows, None))
        except Exception as exc:
            self._results.put(([], f"Unable to load attendance: {exc}"))

    def _poll(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            rows, error = self._results.get_nowait()
        except Empty:
            self._poll_id = self.after(50, self._poll)
            return
        self._busy = False
        for row in rows:
            self.table.insert("", "end", values=row)
        self.refresh_button.configure(state="normal")
        self.filter_button.configure(state="normal")
        scope = "all dates" if self.show_all.get() else "today"
        self.status_label.configure(text=error or (
            f"{len(rows)} record(s) — {scope}. Times: Asia/Saigon (UTC+07)." if rows
            else f"No attendance records for {scope}."
        ))

    def destroy(self):
        self._closed = True
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        super().destroy()
