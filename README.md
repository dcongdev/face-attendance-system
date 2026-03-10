# Face Recognition Attendance System

A Python desktop application for automated student attendance management using face recognition.

## Features

- Real-time face detection and recognition through webcam.
- Student registration using FaceNet face embeddings.
- Student profile management, including editing, updating face data, and deletion.
- Automatic attendance recording with duplicate check-in prevention.
- Attendance history with date filtering and record management.
- Visual status indicators for recognition and attendance results.
- Persistent student and attendance data using SQLite.
- Desktop GUI built with Tkinter.

## Tech Stack

- **Language:** Python
- **GUI:** Tkinter
- **Computer Vision:** OpenCV
- **Face Recognition:** DeepFace (FaceNet)
- **Numerical Computing:** NumPy
- **Database:** SQLite
- **Environment & Dependencies:** uv

## Getting Started

**Requirements:** Python 3.11, uv, and a working webcam.

Clone the repository:

```bash
git clone https://github.com/dcongdev/face-attendance-system.git
cd face-attendance-system
```

Install dependencies:

```bash
uv sync
```

Run the application:

```bash
uv run python main.py
```

The pretrained FaceNet model may be downloaded automatically on first use.

## How It Works

1. Register a student using their ID, name, and webcam.
2. FaceNet extracts and stores a face embedding for recognition.
3. The application recognizes registered students through the webcam.
4. Enable attendance mode to automatically record check-ins.
5. View and manage attendance history and student profiles.

## Notes

- Face recognition uses pretrained FaceNet embeddings and similarity matching.
- Each student can check in once per day.
- Local databases and biometric data are excluded from version control.