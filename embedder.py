"""
embedder.py — Lightweight face embeddings using MediaPipe + HOG features.
No tensorflow, no dlib, no cmake. Works on Streamlit Cloud free tier.
"""

import cv2
import numpy as np


def get_embedder():
    """Returns a simple dict config — no heavy model to load."""
    return {"ready": True}


def get_embedding(embedder, rgb_img: np.ndarray):
    """
    Extract a lightweight 512-d feature vector from a face crop.
    Uses HOG (histogram of oriented gradients) + LBP (local binary pattern)
    — fast, CPU-only, no external model files needed.
    """
    try:
        # Resize to standard size
        face = cv2.resize(rgb_img, (64, 64))
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        # ── HOG features ──────────────────────────────────────────
        win_size    = (64, 64)
        block_size  = (16, 16)
        block_stride= (8, 8)
        cell_size   = (8, 8)
        nbins       = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_feat = hog.compute(gray).flatten()          # 1764-d

        # ── LBP features ──────────────────────────────────────────
        lbp = _lbp(gray)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
        hist_lbp    = hist_lbp.astype(np.float32)

        # ── Color histogram (HSV) ─────────────────────────────────
        hsv  = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()

        # ── Concatenate + L2 normalise ────────────────────────────
        feat = np.concatenate([hog_feat, hist_lbp, h_hist, s_hist])
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat.astype(np.float32)

    except Exception:
        return None


def _lbp(gray: np.ndarray) -> np.ndarray:
    """Compute uniform Local Binary Pattern image."""
    h, w   = gray.shape
    lbp    = np.zeros_like(gray)
    g      = gray.astype(np.int32)
    neighbors = [
        (-1,-1),(-1,0),(-1,1),
        ( 0, 1),( 1, 1),( 1, 0),
        ( 1,-1),( 0,-1),
    ]
    for i, (dy, dx) in enumerate(neighbors):
        shifted = np.zeros_like(g)
        sy = slice(max(0,-dy), h + min(0,-dy))
        sx = slice(max(0,-dx), w + min(0,-dx))
        ty = slice(max(0, dy), h + min(0, dy))
        tx = slice(max(0, dx), w + min(0, dx))
        shifted[ty, tx] = g[sy, sx]
        lbp += ((shifted >= g).astype(np.uint8) << i)
    return lbp.astype(np.uint8)
