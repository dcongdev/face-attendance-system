from datetime import date, datetime
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk
from gui.date_picker import DatePicker

from database.db import (
    ATTENDANCE_TIMEZONE, DEFAULT_DB_PATH, get_attendance_history, initialize_database,
)


def _load_history(db_path, target_date, results):
    try:
        initialize_database(db_path)
        records = get_attendance_history(db_path)
        # Explicitly order by attendance day, then actual check-in instant.
        records.sort(key=lambda record: record.student_id)
        records.sort(key=lambda record: (
            record.attendance_date, datetime.fromisoformat(record.checked_in_at).timestamp()
        ), reverse=True)
        rows = [
            (record.student_id, record.full_name, record.attendance_date,
             datetime.fromisoformat(record.checked_in_at).astimezone(ATTENDANCE_TIMEZONE).strftime("%H:%M:%S"))
            for record in records if target_date is None or record.attendance_date == target_date
        ]
        results.put((rows, None))
    except Exception as exc:
        results.put(([], f"Unable to load attendance: {exc}"))


class AttendanceHistoryWindow(tk.Toplevel):
    """Read-only attendance table with background refresh and a local-day filter."""

    def __init__(self, master, db_path=DEFAULT_DB_PATH):
        super().__init__(master)
        self.title("Attendance History")
        self.geometry("900x500")
        self.minsize(720, 360)
        self.db_path = db_path
        self._results = Queue()
        self._busy = False
        self._closed = False
        self._poll_id = None
        self.filter_mode = tk.StringVar(self, value="today")
        self.selected_date = tk.StringVar(self, value=datetime.now(ATTENDANCE_TIMEZONE).date().isoformat())
        self._calendar_window = None
        self._scope = "today"

        content = ttk.Frame(self, padding=15)
        content.pack(fill="both", expand=True)
        controls = ttk.Frame(content)
        controls.pack(fill="x", pady=(0, 10))
        self.filter_buttons = {}
        for mode, label in (("today", "Today"), ("date", "Select Date"), ("all", "All Records")):
            button = ttk.Radiobutton(controls, text=label, variable=self.filter_mode,
                                     value=mode, command=self.refresh)
            button.pack(side="left", padx=(0, 15))
            self.filter_buttons[mode] = button
        self.refresh_button = ttk.Button(controls, text="Refresh", command=self.refresh)
        self.refresh_button.pack(side="right")
        date_controls = ttk.Frame(content)
        date_controls.pack(fill="x", pady=(0, 12))
        ttk.Label(date_controls, text="Date (YYYY-MM-DD):").pack(side="left", padx=(0, 8))
        self.date_entry = ttk.Entry(date_controls, textvariable=self.selected_date, width=14)
        self.date_entry.pack(side="left")
        self.date_entry.bind("<Return>", lambda event: self.refresh())
        self.calendar_button = ttk.Button(date_controls, text="Choose Date…", command=self.choose_date)
        self.calendar_button.pack(side="left", padx=8)
        self.count_label = ttk.Label(date_controls, text="Records: 0")
        self.count_label.pack(side="right")

        table_frame = ttk.Frame(content)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(
            table_frame, columns=("student_id", "full_name", "date", "time"),
            show="headings", selectmode="extended",
        )
        for column, title, width in (
            ("student_id", "Student ID", 120), ("full_name", "Full Name", 260),
            ("date", "Date", 130), ("time", "Check-in Time", 150),
        ):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, minwidth=110,
                              anchor="w" if column in ("student_id", "full_name") else "center",
                              stretch=column == "full_name")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(content, text="", wraplength=600)
        self.status_label.pack(fill="x", pady=(10, 0))
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()

    def choose_date(self):
        if self._busy or self.filter_mode.get() != "date":
            return
        if self._calendar_window is not None and self._calendar_window.winfo_exists():
            self._calendar_window.lift()
            return
        try:
            selected = date.fromisoformat(self.selected_date.get())
        except ValueError:
            selected = datetime.now(ATTENDANCE_TIMEZONE).date()
        self._calendar_window = DatePicker(self, selected, self._select_date)

    def _select_date(self, selected):
        self.selected_date.set(selected.isoformat())
        self.refresh()

    def _update_controls(self):
        state = "disabled" if self._busy else "normal"
        self.refresh_button.configure(state=state)
        for button in self.filter_buttons.values():
            button.configure(state=state)
        date_state = "normal" if not self._busy and self.filter_mode.get() == "date" else "disabled"
        self.date_entry.configure(state=date_state)
        self.calendar_button.configure(state=date_state)

    def refresh(self):
        if self._closed or self._busy:
            return
        mode = self.filter_mode.get()
        try:
            target_date = (date.fromisoformat(self.selected_date.get()).isoformat() if mode == "date"
                           else datetime.now(ATTENDANCE_TIMEZONE).date().isoformat())
        except ValueError:
            self._update_controls()
            self.status_label.configure(text="Invalid date. Choose a calendar date or enter YYYY-MM-DD.")
            return
        self._scope = "all dates" if mode == "all" else target_date
        self._busy = True
        self._update_controls()
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        self.status_label.configure(text="Loading attendance...")
        self.count_label.configure(text="Records: 0")
        Thread(target=_load_history,
               args=(self.db_path, None if mode == "all" else target_date, self._results),
               daemon=True).start()
        self._poll_id = self.after(50, self._poll)

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
        self._update_controls()
        self.count_label.configure(text=f"Records: {len(rows)}")
        scope = self._scope
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
        # Release Tcl variables on the GUI thread, even if background loading continues.
        self.filter_mode = None
        self.selected_date = None
