import os
import pickle
import time

import cv2
import numpy as np
import streamlit as st

from augment import augment_face
from embedder import get_embedder, get_embedding
from classifier import train_classifier, predict_face

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "embeddings")
MODEL_PATH = os.path.join(BASE_DIR, "data", "classifier.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

FRAME_SAMPLE = 5

st.set_page_config(page_title="Face Recognizer", page_icon="🎭", layout="centered")
st.title("🎭 Face Recognizer")
st.caption("Lightweight face recognition — no GPU needed.")

if "embedder"    not in st.session_state: st.session_state.embedder    = get_embedder()
if "classifier"  not in st.session_state: st.session_state.classifier  = None
if "label_names" not in st.session_state: st.session_state.label_names = []

def load_classifier():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            d = pickle.load(f)
        st.session_state.classifier  = d["clf"]
        st.session_state.label_names = d["labels"]

def save_classifier(clf, labels):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"clf": clf, "labels": labels}, f)
    st.session_state.classifier  = clf
    st.session_state.label_names = labels

if st.session_state.classifier is None:
    load_classifier()

with st.sidebar:
    st.header("Known persons")
    persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if persons:
        for p in persons:
            st.success(p.replace("_", " "))
    else:
        st.info("No persons added yet.")
    st.caption("Add at least 2 persons before recognition.")

tab1, tab2 = st.tabs(["Add Person", "Live Recognition"])

with tab1:
    st.subheader("Register a new person")
    name_input = st.text_input("Enter person name", placeholder="e.g. Alice")
    record_dur = st.slider("Recording duration (seconds)", 20, 120, 60, 10)

    if st.button("Start Recording", disabled=not name_input.strip(), use_container_width=True):
        name = name_input.strip().replace(" ", "_")
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.stop()

        frame_ph  = st.empty()
        prog_ph   = st.empty()
        status_ph = st.empty()

        embeddings = []
        idx = 0
        t0  = time.time()

        while True:
            elapsed = time.time() - t0
            if elapsed >= record_dur:
                break
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, channels="RGB",
                           caption=f"{int(record_dur - elapsed)}s left",
                           use_container_width=True)
            prog_ph.progress(min(elapsed / record_dur, 1.0))

            if idx % FRAME_SAMPLE != 0:
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                crop = frame[y:y+h, x:x+w]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                for v in augment_face(crop_rgb):
                    e = get_embedding(st.session_state.embedder, v)
                    if e is not None:
                        embeddings.append(e)

            status_ph.info(f"Collecting... {len(embeddings)} embeddings so far")

        cap.release()
        frame_ph.empty()
        prog_ph.empty()

        if not embeddings:
            status_ph.error("No faces detected. Try better lighting.")
        else:
            path = os.path.join(DATA_DIR, name, "embeddings.pkl")
            with open(path, "wb") as f:
                pickle.dump(np.array(embeddings), f)

            clf, labels = train_classifier(DATA_DIR)
            if clf:
                save_classifier(clf, labels)
                status_ph.success(
                    f"Done! {len(embeddings)} embeddings saved for '{name}'. "
                    f"Classifier trained on {len(labels)} person(s).")
            else:
                status_ph.warning(f"Saved '{name}'. Add one more person to train.")
            st.rerun()

with tab2:
    st.subheader("Live recognition")

    if st.session_state.classifier is None:
        st.warning("Add at least 2 persons first.")
    else:
        st.success(f"Model ready — knows: {', '.join(st.session_state.label_names)}")
        threshold = st.slider("Confidence threshold", 0.2, 0.9, 0.55, 0.05)
        run = st.toggle("Start camera")

        if run:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
                st.stop()

            frame_ph = st.empty()
            st.info("Toggle off to stop.")

            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

                for (x, y, w, h) in faces:
                    crop = frame[y:y+h, x:x+w]
                    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    emb  = get_embedding(st.session_state.embedder, rgb)

                    if emb is not None:
                        name, conf = predict_face(
                            st.session_state.classifier,
                            st.session_state.label_names,
                            emb, threshold)
                    else:
                        name, conf = "Unknown", 0.0

                    color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({conf:.0%})",
                                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
            cap.release()
