from datetime import date, datetime
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import messagebox, ttk
from gui.date_picker import DatePicker

from database.db import (
    ATTENDANCE_TIMEZONE, DEFAULT_DB_PATH, get_attendance_history, initialize_database,
    delete_attendance_records,
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
            (record.record_id, (record.student_id, record.full_name, record.attendance_date,
             datetime.fromisoformat(record.checked_in_at).astimezone(ATTENDANCE_TIMEZONE).strftime("%H:%M:%S")))
            for record in records if target_date is None or record.attendance_date == target_date
        ]
        results.put((rows, None))
    except Exception as exc:
        results.put(([], f"Unable to load attendance: {exc}"))


def _prepare_deletion(db_path, target_date, selected_ids, results):
    try:
        records = get_attendance_history(db_path)
        ids = [record.record_id for record in records
               if (target_date is None or record.attendance_date == target_date)
               and (selected_ids is None or record.record_id in selected_ids)]
        results.put(("confirm", ids, None))
    except Exception as exc:
        results.put(("confirm", [], f"Unable to prepare deletion: {exc}"))


def _delete_history(db_path, ids, results):
    try:
        count = delete_attendance_records(ids, db_path)
        results.put(("deleted", count, None))
    except Exception as exc:
        results.put(("deleted", 0, f"Unable to delete attendance: {exc}"))


class AttendanceHistoryWindow(tk.Toplevel):
    """Attendance table with background refresh, date filters and confirmed deletion."""

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
        self._delete_results = Queue()
        self._delete_scope = ""
        self._delete_feedback = ""

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
        actions = ttk.Frame(content)
        actions.pack(fill="x", pady=(10, 0))
        self.delete_buttons = {}
        for action, label in (("selected", "Delete Selected"), ("filtered", "Clear Filtered"), ("all", "Clear All")):
            button = ttk.Button(actions, text=label, command=lambda action=action: self.request_deletion(action))
            button.pack(side="left", padx=(0, 10))
            self.delete_buttons[action] = button
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
        for button in self.delete_buttons.values():
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
        for record_id, row in rows:
            self.table.insert("", "end", iid=str(record_id), values=row)
        self._update_controls()
        self.count_label.configure(text=f"Records: {len(rows)}")
        scope = self._scope
        self.status_label.configure(text=error or self._delete_feedback + (
            f"{len(rows)} record(s) — {scope}. Times: Asia/Saigon (UTC+07)." if rows
            else f"No attendance records for {scope}."
        ))
        self._delete_feedback = ""

    def request_deletion(self, action):
        if self._closed or self._busy:
            return
        selected_ids = None
        if action == "selected":
            selected_ids = {int(item) for item in self.table.selection()}
            if not selected_ids:
                self.status_label.configure(text="Select one or more attendance records to delete.")
                return
            scope, target_date = "selected attendance records", None
        elif action == "all":
            scope, target_date = "ENTIRE attendance history (all dates)", None
        elif action == "filtered":
            try:
                mode = self.filter_mode.get()
                target_date = (None if mode == "all" else
                               date.fromisoformat(self.selected_date.get()).isoformat() if mode == "date"
                               else datetime.now(ATTENDANCE_TIMEZONE).date().isoformat())
            except ValueError:
                self.status_label.configure(text="Invalid date. Choose a valid date before clearing records.")
                return
            scope = f"attendance on {target_date}" if target_date else "all dates (All Records filter)"
        else:
            return
        self._delete_scope = scope
        self._busy = True
        self._update_controls()
        self.status_label.configure(text="Checking records to delete...")
        Thread(target=_prepare_deletion,
               args=(self.db_path, target_date, selected_ids, self._delete_results), daemon=True).start()
        self._poll_id = self.after(50, self._poll_deletion)

    def _poll_deletion(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            stage, value, error = self._delete_results.get_nowait()
        except Empty:
            self._poll_id = self.after(50, self._poll_deletion)
            return
        if error:
            self._busy = False
            self._update_controls()
            self.status_label.configure(text=error)
            return
        if stage == "confirm":
            if not value:
                self._busy = False
                self._update_controls()
                self.status_label.configure(text=f"No attendance records to delete: {self._delete_scope}.")
                return
            confirmed = messagebox.askyesno(
                "Delete Attendance Records",
                f"Delete {len(value)} attendance record(s)?\nScope: {self._delete_scope}.\n\n"
                "Only these confirmed records will be deleted; later check-ins are preserved.\n"
                "Student profiles and face embeddings will be kept.", parent=self, default="no",
            )
            if self._closed:
                return
            if not confirmed:
                self._busy = False
                self._update_controls()
                self.status_label.configure(text="Deletion cancelled.")
                return
            self.status_label.configure(text="Deleting attendance records...")
            Thread(target=_delete_history, args=(self.db_path, value, self._delete_results), daemon=True).start()
            self._poll_id = self.after(50, self._poll_deletion)
        else:
            self._busy = False
            self._delete_feedback = f"Deleted {value} record(s). "
            self.refresh()

    def destroy(self):
        self._closed = True
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        super().destroy()
        # Release Tcl variables on the GUI thread, even if background loading continues.
        self.filter_mode = None
        self.selected_date = None
