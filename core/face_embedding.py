import numpy as np


def extract_embedding(frame):
    """Return a FaceNet embedding only when exactly one face is detected in a BGR frame."""
    if frame is None or frame.size == 0:
        raise ValueError("No camera frame available. Start the camera and try again.")

    # Import and model initialization happen in the inference worker, not during GUI startup.
    from deepface import DeepFace

    try:
        faces = DeepFace.extract_faces(
            img_path=frame, detector_backend="opencv", enforce_detection=True,
            align=True, color_face="bgr", normalize_face=False,
        )
    except ValueError as exc:
        if "face could not be detected" in str(exc).lower():
            raise ValueError("No face detected. Face the camera and try again.") from exc
        raise
    if not faces:
        raise ValueError("No face detected. Face the camera and try again.")
    if len(faces) != 1:
        raise ValueError("Multiple faces detected. Keep only one face in view.")
    result = DeepFace.represent(
        img_path=faces[0]["face"], model_name="Facenet",
        detector_backend="skip", enforce_detection=True,
    )
    embedding = np.asarray(result[0]["embedding"], dtype=np.float32)
    if embedding.shape != (128,) or not np.isfinite(embedding).all():
        raise ValueError("FaceNet returned an invalid embedding.")
    return embedding
