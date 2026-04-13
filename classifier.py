"""
classifier.py — Train an SVM on stored face embeddings and run prediction.
"""

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def train_classifier(data_dir: str):
    """
    Scan `data_dir` for per-person embedding files, build an SVM classifier,
    and return (clf_pipeline, label_names_list).

    Returns (None, []) if fewer than 2 people are in the DB.
    """
    X, y = [], []

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        emb_path   = os.path.join(person_dir, "embeddings.pkl")
        if not os.path.isfile(emb_path):
            continue
        with open(emb_path, "rb") as f:
            embeddings = pickle.load(f)          # numpy array (N, 128)
        for emb in embeddings:
            X.append(emb)
            y.append(person_name)

    if len(set(y)) < 2:
        return None, []

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # L2-normalise embeddings then train a probabilistic SVM
    clf = Pipeline([
        ("norm", Normalizer(norm="l2")),
        ("svm",  SVC(kernel="rbf", C=10.0, gamma="scale",
                     probability=True, class_weight="balanced")),
    ])
    clf.fit(X, y_enc)

    return clf, list(le.classes_)


def predict_face(clf, label_names: list, embedding: np.ndarray,
                 threshold: float = 0.45):
    """
    Predict the identity for a single 128-d embedding.

    Returns (name, confidence) where name is "Unknown" if the top-class
    probability is below `threshold`.
    """
    emb  = embedding.reshape(1, -1)
    probs = clf.predict_proba(emb)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])

    if conf < threshold:
        return "Unknown", conf

    return label_names[idx], conf
