import tkinter as tk
from tkinter import ttk


class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Face Recognition Attendance System")
        self.geometry("1000x650")
        self.minsize(800, 550)

        self.create_widgets()

    def create_widgets(self):
        # Title
        title = ttk.Label(
            self,
            text="Face Recognition Attendance System",
            font=("Arial", 20, "bold"),
        )
        title.pack(pady=20)

        # Main content
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Webcam preview
        camera_frame = ttk.LabelFrame(
            main_frame,
            text="Webcam Preview",
            padding=10,
        )
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.camera_label = ttk.Label(
            camera_frame,
            text="Camera is not started",
            anchor="center",
        )
        self.camera_label.pack(fill="both", expand=True)

        # Control panel
        control_frame = ttk.LabelFrame(
            main_frame,
            text="Student Management",
            padding=15,
        )
        control_frame.pack(side="right", fill="y")

        ttk.Label(control_frame, text="Student ID").pack(anchor="w")
        self.student_id_entry = ttk.Entry(control_frame, width=30)
        self.student_id_entry.pack(pady=(5, 15))

        ttk.Label(control_frame, text="Full Name").pack(anchor="w")
        self.student_name_entry = ttk.Entry(control_frame, width=30)
        self.student_name_entry.pack(pady=(5, 20))

        ttk.Button(
            control_frame,
            text="Register Student",
        ).pack(fill="x", pady=5)

        ttk.Button(
            control_frame,
            text="Start Attendance",
        ).pack(fill="x", pady=5)

        ttk.Button(
            control_frame,
            text="Stop Attendance",
        ).pack(fill="x", pady=5)

        # Status bar
        self.status_label = ttk.Label(
            self,
            text="Status: Ready",
            padding=10,
        )
        self.status_label.pack(side="bottom", fill="x")