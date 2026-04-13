 """
streamlit_app.py — Face Recognizer (lightweight, Streamlit Cloud compatible)
Uses HOG + LBP features — no dlib, no tensorflow, no cmake needed.
Run: streamlit run streamlit_app.py
"""

import os
import pickle
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from augment import augment_face
from embedder import get_embedder, get_embedding
from classifier import train_classifier, predict_face

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "embeddings")
MODEL_PATH = os.path.join(BASE_DIR, "data", "classifier.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

RECORD_SECONDS = 180
FRAME_SAMPLE   = 5

st.set_page_config(page_title="Face Recognizer", page_icon="🎭", layout="centered")
st.title("🎭 Face Recognizer")

# ── Session state ──────────────────────────────────────────────────────────────
if "embedder"    not in st.session_state:
    st.session_state.embedder    = get_embedder()   # lightweight, instant
if "classifier"  not in st.session_state:
    st.session_state.classifier  = None
if "label_names" not in st.session_state:
    st.session_state.label_names = []

# ── Load saved classifier ──────────────────────────────────────────────────────
def load_classifier():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        st.session_state.classifier  = data["clf"]
        st.session_state.label_names = data["labels"]

def save_classifier(clf, labels):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"clf": clf, "labels": labels}, f)
    st.session_state.classifier  = clf
    st.session_state.label_names = labels

if st.session_state.classifier is None:
    load_classifier()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Known persons")
    persons = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))]
    if persons:
        for p in persons:
            st.markdown(f"✅ {p.replace('_', ' ')}")
    else:
        st.info("No persons added yet.")
    st.divider()
    st.caption("Add ≥ 2 persons before recognition.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["➕ Add Person", "▶ Live Recognition"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Add Person
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Register a new person from webcam video")

    name_input = st.text_input("Person's name", placeholder="e.g. Alice")

    col1, col2 = st.columns(2)
    record_duration = col1.slider("Recording duration (seconds)", 30, 180, 60, 10)
    frame_skip      = col2.slider("Sample every N frames", 1, 10, 5)

    start_btn = st.button("🎥 Start Recording", use_container_width=True,
                          disabled=not name_input.strip())

    cam_placeholder  = st.empty()
    prog_placeholder = st.empty()
    status_box       = st.empty()

    if start_btn and name_input.strip():
        name = name_input.strip().replace(" ", "_")

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        person_dir = os.path.join(DATA_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Make sure no other app is using it.")
            st.stop()

        embeddings = []
        frame_idx  = 0
        saved      = 0
        start_time = time.time()

        status_box.info(f"Recording '{name}' — move your face slowly in all directions …")

        while True:
            elapsed = time.time() - start_time
            if elapsed >= record_duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb_preview, channels="RGB",
                                  caption=f"Recording… {int(record_duration - elapsed)}s left",
                                  use_container_width=True)
            prog_placeholder.progress(min(elapsed / record_duration, 1.0))

            if frame_idx % frame_skip != 0:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                variants  = augment_face(face_rgb)

                for variant in variants:
                    emb = get_embedding(st.session_state.embedder, variant)
                    if emb is not None:
                        embeddings.append(emb)
                        saved += 1

            status_box.info(
                f"Recording '{name}' — {int(record_duration - elapsed)}s left | "
                f"embeddings: {saved}")

        cap.release()
        cam_placeholder.empty()
        prog_placeholder.empty()

        if not embeddings:
            status_box.error("No faces detected! Try better lighting.")
        else:
            emb_path = os.path.join(person_dir, "embeddings.pkl")
            with open(emb_path, "wb") as f:
                pickle.dump(np.array(embeddings), f)

            status_box.success(f"Saved {len(embeddings)} embeddings for '{name}'. Training …")

            clf, labels = train_classifier(DATA_DIR)
            if clf is not None:
                save_classifier(clf, labels)
                status_box.success(
                    f"✅ Done! Classifier ready for {len(labels)} person(s). "
                    "Switch to **Live Recognition** tab.")
            else:
                status_box.warning("Add at least 2 persons to train the classifier.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Recognition
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live face recognition")

    if st.session_state.classifier is None:
        st.warning("No classifier trained yet. Add at least 2 persons first.")
    else:
        st.success(f"Ready — knows: {', '.join(st.session_state.label_names)}")

        threshold = st.slider("Confidence threshold", 0.20, 0.90, 0.55, 0.05,
                              help="Faces below this score are labelled Unknown")

        run_recog  = st.toggle("Run live recognition", value=False)
        live_frame = st.empty()

        if run_recog:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
                st.stop()

            st.info("Recognition running — toggle off to stop.")

            while run_recog:
                ret, frame = cap.read()
                if not ret:
                    break

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    face_crop = frame[y:y+h, x:x+w]
                    face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    emb       = get_embedding(st.session_state.embedder, face_rgb)

                    if emb is not None:
                        pred_name, conf = predict_face(
                            st.session_state.classifier,
                            st.session_state.label_names,
                            emb, threshold)
                    else:
                        pred_name, conf = "Unknown", 0.0

                    color = (0, 200, 0) if pred_name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{pred_name} ({conf:.0%})",
                                (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                live_frame.image(rgb, channels="RGB", use_container_width=True)

            cap.release()
