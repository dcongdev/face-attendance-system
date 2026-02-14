import tkinter as tk
from tkinter import ttk
from queue import Empty

import cv2
from PIL import Image, ImageTk

from gui.camera import CameraCapture
from gui.window_state import load_geometry, save_geometry


class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Face Recognition Attendance System")
        self.geometry(load_geometry(self))
        self.minsize(800, 550)

        self.camera = None
        self._preview_image = None
        self._stopping = False
        self._closing = False
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.close)

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

        camera_controls = ttk.Frame(camera_frame)
        camera_controls.pack(side="bottom", fill="x", pady=(10, 0))
        self.start_camera_button = ttk.Button(
            camera_controls, text="Start Camera", command=self.start_camera
        )
        self.start_camera_button.pack(side="left", padx=(0, 10))
        self.stop_camera_button = ttk.Button(
            camera_controls, text="Stop Camera", command=self.stop_camera,
            state="disabled",
        )
        self.stop_camera_button.pack(side="left")

        preview_frame = ttk.Frame(camera_frame)
        preview_frame.pack(fill="both", expand=True)
        preview_frame.pack_propagate(False)

        self.camera_label = ttk.Label(
            preview_frame,
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

    def start_camera(self):
        if self.camera is not None or self._closing:
            return
        self._stopping = False
        self.camera = CameraCapture()
        self.start_camera_button.configure(state="disabled")
        self.stop_camera_button.configure(state="normal")
        self.status_label.configure(text="Status: Opening camera...")
        self.camera.start()
        self.after(30, self._update_preview)

    def stop_camera(self):
        if self.camera is None:
            return
        self._stopping = True
        self.camera.stop()
        self.stop_camera_button.configure(state="disabled")
        self.status_label.configure(text="Status: Stopping camera...")
        self._clear_preview()

    def _clear_preview(self):
        self.camera_label.configure(image="", text="Camera is not started")
        self._preview_image = None

    def _update_preview(self):
        try:
            kind, value = self.camera.messages.get_nowait()
        except Empty:
            pass
        else:
            if kind == "stopped":
                self.camera = None
                self._clear_preview()
                if self._closing:
                    self.destroy()
                    return
                self.start_camera_button.configure(state="normal")
                self.stop_camera_button.configure(state="disabled")
                self.status_label.configure(
                    text=f"Status: {value}" if value else "Status: Camera stopped"
                )
                return
            if not self._stopping:
                try:
                    image = Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
                    image.thumbnail((max(1, self.camera_label.winfo_width()),
                                     max(1, self.camera_label.winfo_height())))
                    self._preview_image = ImageTk.PhotoImage(image)
                    self.camera_label.configure(image=self._preview_image, text="")
                    self.status_label.configure(text="Status: Camera running")
                except (cv2.error, ValueError, tk.TclError):
                    self.stop_camera()
        self.after(30, self._update_preview)

    def close(self):
        if not self._closing:
            save_geometry(self)
        self._closing = True
        if self.camera is None:
            self.destroy()
        else:
            # Wait via after() for the worker to release its handle, without joining Tk.
            self.stop_camera()
