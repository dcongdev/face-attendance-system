import tkinter as tk
from tkinter import ttk
from queue import Empty, Queue
from threading import Thread

import cv2
from PIL import Image, ImageTk

from gui.camera import CameraCapture
from gui.window_state import load_geometry, save_geometry
from core.face_embedding import extract_embedding
from core.registration import RegistrationResult, register_student


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
        self._embedding_results = Queue()
        self._embedding_busy = False
        self._embedding_status = None
        self._registration_results = Queue()
        self._registration_busy = False
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
        self.extract_button = ttk.Button(
            camera_controls, text="Extract Embedding", command=self.extract_embedding
        )
        self.extract_button.pack(side="left", padx=(10, 0))

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

        self.register_button = ttk.Button(
            control_frame,
            text="Register Student",
            command=self.register_student,
        )
        self.register_button.pack(fill="x", pady=5)

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
        self._embedding_status = None
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
        self._embedding_status = None
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
                    if not self._registration_busy:
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
                    self.status_label.configure(
                        text=self._embedding_status or "Status: Camera running"
                    )
                except (cv2.error, ValueError, tk.TclError):
                    self.stop_camera()
        self.after(30, self._update_preview)

    def extract_embedding(self):
        if self._embedding_busy or self._registration_busy or self._closing:
            return
        frame = self.camera.snapshot() if self.camera and not self._stopping else None
        if frame is None:
            self._embedding_status = "Status: No camera frame available. Start the camera first."
            self.status_label.configure(text=self._embedding_status)
            return
        self._embedding_busy = True
        self.extract_button.configure(state="disabled")
        self.register_button.configure(state="disabled")
        self._embedding_status = "Status: Extracting embedding (first run may take longer)..."
        self.status_label.configure(text=self._embedding_status)
        Thread(target=self._extract_in_background, args=(frame, self.camera), daemon=True).start()
        self.after(50, self._poll_embedding)

    def _extract_in_background(self, frame, camera):
        try:
            embedding = extract_embedding(frame)
            message = f"Status: FaceNet embedding extracted: {embedding.size} dimensions"
        except Exception as exc:
            message = f"Status: Embedding error: {exc}"
        self._embedding_results.put((camera, message))

    def _poll_embedding(self):
        if self._closing:
            return
        try:
            camera, message = self._embedding_results.get_nowait()
        except Empty:
            self.after(50, self._poll_embedding)
            return
        self._embedding_busy = False
        self.extract_button.configure(state="normal")
        self.register_button.configure(state="normal")
        if camera is self.camera and not self._stopping:
            self._embedding_status = message
            self.status_label.configure(text=message)

    def register_student(self):
        if self._registration_busy or self._embedding_busy or self._closing:
            return
        student_id = self.student_id_entry.get().strip()
        full_name = self.student_name_entry.get().strip()
        if not student_id or student_id == "Unknown":
            self._show_registration_status("Enter a student ID other than 'Unknown'.")
            return
        if not full_name:
            self._show_registration_status("Enter the student's full name.")
            return
        frame = self.camera.snapshot() if self.camera and not self._stopping else None
        if frame is None:
            self._show_registration_status("No camera frame available. Start the camera first.")
            return
        self._registration_busy = True
        self.register_button.configure(state="disabled")
        self.extract_button.configure(state="disabled")
        self.student_id_entry.configure(state="disabled")
        self.student_name_entry.configure(state="disabled")
        self._show_registration_status(f"Registering {student_id!r} (first run may take longer)...")
        Thread(target=self._register_in_background,
               args=(student_id, full_name, frame), daemon=True).start()
        self.after(50, self._poll_registration)

    def _show_registration_status(self, message):
        self._embedding_status = f"Status: {message}"
        self.status_label.configure(text=self._embedding_status)

    def _register_in_background(self, student_id, full_name, frame):
        try:
            result = register_student(student_id, full_name, frame)
        except Exception as exc:
            result = RegistrationResult(False, "registration_error", f"Registration failed: {exc}")
        self._registration_results.put(result)

    def _poll_registration(self):
        try:
            result = self._registration_results.get_nowait()
        except Empty:
            self.after(50, self._poll_registration)
            return
        self._registration_busy = False
        if self._closing:
            if self.camera is None:
                self.destroy()
            return
        self.register_button.configure(state="normal")
        self.extract_button.configure(state="normal")
        self.student_id_entry.configure(state="normal")
        self.student_name_entry.configure(state="normal")
        self._show_registration_status(result.message)

    def close(self):
        if not self._closing:
            save_geometry(self)
        self._closing = True
        if self.camera is None and not self._registration_busy:
            self.destroy()
        elif self.camera is not None:
            # Wait via after() for the worker to release its handle, without joining Tk.
            self.stop_camera()
