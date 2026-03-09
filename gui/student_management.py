from datetime import datetime
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk
from core.student_profile import update_student_face

from database.db import (
    ATTENDANCE_TIMEZONE, DEFAULT_DB_PATH, initialize_database, search_students, update_student_name,
)


def _load_profiles(db_path, query, results):
    try:
        initialize_database(db_path)
        profiles = search_students(query, db_path)
        rows = [(p.student_id, p.full_name,
                 datetime.fromisoformat(p.registered_at).astimezone(ATTENDANCE_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S"))
                for p in profiles]
        results.put(("load", rows, None))
    except Exception as exc:
        results.put(("load", [], f"Unable to load students: {exc}"))


def _save_profile(db_path, student_id, full_name, results):
    try:
        results.put(("save", update_student_name(student_id, full_name, db_path), None))
    except Exception as exc:
        results.put(("save", None, f"Unable to save student: {exc}"))


def _update_face(db_path, student_id, frame, results):
    try:
        results.put(("face", update_student_face(student_id, frame, db_path), None))
    except Exception as exc:
        results.put(("face", None, f"Unable to update face: {exc}"))


class StudentManagementWindow(tk.Toplevel):
    """Search profiles, edit names and replace face embeddings using webcam snapshots."""

    def __init__(self, master, db_path=DEFAULT_DB_PATH, on_updated=None, capture_frame=None):
        super().__init__(master)
        self.title("Student Management")
        self.geometry("800x500")
        self.minsize(650, 380)
        self.db_path = db_path
        self.on_updated = on_updated
        self.capture_frame = capture_frame
        self._results = Queue()
        self._busy = False
        self._closed = False
        self._poll_id = None
        self._selected_id = None
        self._restore_id = None
        self._feedback = ""
        self.query = tk.StringVar(self)
        self.student_id = tk.StringVar(self)
        self.full_name = tk.StringVar(self)

        content = ttk.Frame(self, padding=15)
        content.pack(fill="both", expand=True)
        search = ttk.Frame(content)
        search.pack(fill="x", pady=(0, 10))
        ttk.Label(search, text="Student ID or name:").pack(side="left", padx=(0, 8))
        self.search_entry = ttk.Entry(search, textvariable=self.query)
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<Return>", lambda event: self.refresh())
        self.search_button = ttk.Button(search, text="Search", command=self.refresh)
        self.search_button.pack(side="left", padx=8)
        self.refresh_button = ttk.Button(search, text="Refresh", command=self.refresh)
        self.refresh_button.pack(side="left")

        table_frame = ttk.Frame(content)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=("id", "name", "registered"),
                                  show="headings", selectmode="browse")
        for column, label, width in (("id", "Student ID", 130), ("name", "Full Name", 280),
                                     ("registered", "Registration Date", 200)):
            self.table.heading(column, text=label)
            self.table.column(column, width=width, minwidth=100)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._on_selection)

        editor = ttk.LabelFrame(content, text="Edit selected student", padding=10)
        editor.pack(fill="x", pady=10)
        ttk.Label(editor, text="Student ID:").grid(row=0, column=0, sticky="w")
        self.id_entry = ttk.Entry(editor, textvariable=self.student_id, state="readonly", width=22)
        self.id_entry.grid(row=0, column=1, padx=8, pady=4, sticky="w")
        ttk.Label(editor, text="Full Name:").grid(row=1, column=0, sticky="w")
        self.name_entry = ttk.Entry(editor, textvariable=self.full_name)
        self.name_entry.grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        editor.columnconfigure(1, weight=1)
        self.save_button = ttk.Button(editor, text="Save Name", command=self.save_name)
        self.save_button.grid(row=1, column=2, padx=8)
        self.update_face_button = ttk.Button(editor, text="Update Face", command=self.update_face)
        self.update_face_button.grid(row=0, column=2, padx=8)
        ttk.Label(editor, text="Update Face uses the latest webcam frame. Keep exactly one face in view.",
                  wraplength=580).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.count_label = ttk.Label(content, text="Students: 0")
        self.count_label.pack(anchor="w")
        self.status_label = ttk.Label(content, wraplength=730)
        self.status_label.pack(fill="x", pady=(5, 0))
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()

    def _update_controls(self):
        state = "disabled" if self._busy else "normal"
        for widget in (self.search_entry, self.search_button, self.refresh_button):
            widget.configure(state=state)
        editing = "normal" if not self._busy and self._selected_id else "disabled"
        self.name_entry.configure(state=editing)
        self.save_button.configure(state=editing)
        self.update_face_button.configure(state=editing)

    def _on_selection(self, event=None):
        if self._busy or self._closed:
            return
        selection = self.table.selection()
        self._selected_id = selection[0] if selection else None
        self.student_id.set(self._selected_id or "")
        self.full_name.set(self.table.item(selection[0], "values")[1] if selection else "")
        self._update_controls()

    def refresh(self):
        if self._busy or self._closed:
            return
        self._busy = True
        self._selected_id = None
        self.student_id.set("")
        self.full_name.set("")
        self._update_controls()
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        self.count_label.configure(text="Students: 0")
        self.status_label.configure(text="Loading students...")
        Thread(target=_load_profiles, args=(self.db_path, self.query.get(), self._results), daemon=True).start()
        self._poll_id = self.after(50, self._poll)

    def save_name(self):
        if self._busy or self._closed:
            return
        if self._selected_id is None:
            self.status_label.configure(text="Select a student first.")
            return
        name = self.full_name.get().strip()
        if not name:
            self.status_label.configure(text="Enter a full name.")
            return
        self._busy = True
        self._update_controls()
        self.status_label.configure(text="Saving student name...")
        Thread(target=_save_profile,
               args=(self.db_path, self._selected_id, name, self._results), daemon=True).start()
        self._poll_id = self.after(50, self._poll)

    def update_face(self):
        if self._busy or self._closed:
            return
        if self._selected_id is None:
            self.status_label.configure(text="Select a student first.")
            return
        try:
            frame = self.capture_frame() if self.capture_frame else None
            if frame is None:
                raise ValueError("No camera frame available. Start the camera in the main window first.")
        except Exception as exc:
            self.status_label.configure(text=f"Unable to capture face: {exc}")
            return
        self._busy = True
        self._update_controls()
        self.status_label.configure(text=f"Updating face for {self._selected_id}; processing may take a moment...")
        Thread(target=_update_face, args=(self.db_path, self._selected_id, frame, self._results), daemon=True).start()
        self._poll_id = self.after(50, self._poll)

    def _poll(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            action, value, error = self._results.get_nowait()
        except Empty:
            self._poll_id = self.after(50, self._poll)
            return
        self._busy = False
        if error:
            self._update_controls()
            self.status_label.configure(text=error)
            return
        if action in ("save", "face"):
            self._restore_id = value.student_id
            self._feedback = (f"Updated face for {value.student_id}. " if action == "face"
                              else f"Saved name for {value.student_id}. ")
            if self.on_updated:
                self.on_updated()
            self.refresh()
            return
        for row in value:
            self.table.insert("", "end", iid=row[0], values=row)
        if self._restore_id and self.table.exists(self._restore_id):
            self.table.selection_set(self._restore_id)
            self._on_selection()
        self._restore_id = None
        self._update_controls()
        self.count_label.configure(text=f"Students: {len(value)}")
        self.status_label.configure(text=self._feedback + (
            "Select a student to edit their full name. Dates: Asia/Saigon (UTC+07)."
            if value else "No students found."
        ))
        self._feedback = ""

    def destroy(self):
        self._closed = True
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        super().destroy()
        self.query = self.student_id = self.full_name = None
        self.on_updated = None
        self.capture_frame = None
