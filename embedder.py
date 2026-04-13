"""
embedder.py — Generates 128-d face embeddings.

Uses the `face_recognition` library (dlib under the hood) which ships with
pre-trained models — no separate download needed beyond `pip install face_recognition`.

If you prefer FaceNet/ArcFace, swap get_embedder() and get_embedding() with a
deepface or insightface implementation; the rest of the app is unchanged.
"""

import numpy as np


def get_embedder():
    """
    Load and return the face_recognition module (acts as our 'embedder').
    Called once at startup so the dlib models load only once.
    """
    try:
        import face_recognition as fr
        return fr
    except ImportError:
        raise RuntimeError(
            "face_recognition is not installed.\n"
            "Run:  pip install face-recognition\n"
            "(Requires cmake + dlib — see README for details.)"
        )


def get_embedding(embedder, rgb_img: np.ndarray):
    """
    Given a face_recognition module and an RGB numpy array of a face crop,
    return a 128-d numpy vector or None if no face is detected.

    face_recognition.face_encodings() runs detection internally, but since
    we already cropped, we tell it the bounding box covers the whole image.
    """
    h, w = rgb_img.shape[:2]
    # Provide the full-image bounding box so it doesn't re-detect
    known_box = [(0, w, h, 0)]   # top, right, bottom, left
    encodings = embedder.face_encodings(rgb_img, known_face_locations=known_box,
                                        num_jitters=1)
    if encodings:
        return encodings[0]       # 128-d numpy array
    return None
