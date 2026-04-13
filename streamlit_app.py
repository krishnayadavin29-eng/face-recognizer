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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "embeddings")
MODEL_PATH = os.path.join(BASE_DIR, "data", "classifier.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Face Recognizer", page_icon="🎭", layout="centered")
st.title("🎭 Face Recognizer")
st.caption("Upload photos to register a person, then upload a test photo to recognise them.")

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Known persons")
    persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if persons:
        for p in persons:
            st.success(p.replace("_", " "))
    else:
        st.info("No persons added yet.")
    st.divider()
    st.caption("Add at least 2 persons before recognition.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Add Person", "Recognise Face"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Add Person via photo upload
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Register a new person")
    st.info("Upload 5–20 clear face photos of one person. More photos = better accuracy.")

    name_input = st.text_input("Person name", placeholder="e.g. Alice")
    uploaded   = st.file_uploader(
        "Upload face photos (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True)

    if st.button("Register Person", disabled=not (name_input.strip() and uploaded),
                 use_container_width=True):

        name = name_input.strip().replace(" ", "_")
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        embeddings = []
        progress   = st.progress(0)
        status     = st.empty()

        for i, file in enumerate(uploaded):
            img_pil = Image.open(file).convert("RGB")
            img_np  = np.array(img_pil)
            gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

            if len(faces) == 0:
                # Try the whole image as face if detection fails
                faces = [(0, 0, img_np.shape[1], img_np.shape[0])]

            for (x, y, w, h) in faces:
                crop = img_np[y:y+h, x:x+w]
                for variant in augment_face(crop):
                    emb = get_embedding(st.session_state.embedder, variant)
                    if emb is not None:
                        embeddings.append(emb)

            progress.progress((i + 1) / len(uploaded))
            status.info(f"Processed {i+1}/{len(uploaded)} photos — {len(embeddings)} embeddings")

        progress.empty()

        if not embeddings:
            status.error("No faces detected in any photo. Try clearer, well-lit face photos.")
        else:
            path = os.path.join(DATA_DIR, name, "embeddings.pkl")
            with open(path, "wb") as f:
                pickle.dump(np.array(embeddings), f)

            clf, labels = train_classifier(DATA_DIR)
            if clf:
                save_classifier(clf, labels)
                status.success(
                    f"Done! Registered '{name}' with {len(embeddings)} embeddings. "
                    f"Classifier trained on {len(labels)} person(s).")
            else:
                status.warning(f"Saved '{name}'. Add at least one more person to enable recognition.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Recognise face from uploaded photo
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Recognise a face")

    if st.session_state.classifier is None:
        st.warning("Add at least 2 persons first.")
    else:
        st.success(f"Model knows: {', '.join(st.session_state.label_names)}")
        threshold  = st.slider("Confidence threshold", 0.2, 0.9, 0.50, 0.05)
        test_file  = st.file_uploader("Upload a photo to recognise", type=["jpg","jpeg","png"])

        if test_file:
            img_pil = Image.open(test_file).convert("RGB")
            img_np  = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

            if len(faces) == 0:
                st.warning("No face detected in this photo. Try a clearer photo.")
            else:
                result_img = img_bgr.copy()
                results    = []

                for (x, y, w, h) in faces:
                    crop = img_np[y:y+h, x:x+w]
                    emb  = get_embedding(st.session_state.embedder, crop)

                    if emb is not None:
                        name, conf = predict_face(
                            st.session_state.classifier,
                            st.session_state.label_names,
                            emb, threshold)
                    else:
                        name, conf = "Unknown", 0.0

                    color = (0, 180, 0) if name != "Unknown" else (0, 0, 200)
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(result_img, f"{name} ({conf:.0%})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    results.append((name, conf))

                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                         caption="Recognition result", use_container_width=True)

                st.divider()
                for name, conf in results:
                    if name != "Unknown":
                        st.success(f"Recognised: **{name.replace('_',' ')}** — confidence {conf:.0%}")
                    else:
                        st.error(f"Unknown person — confidence too low ({conf:.0%})")
