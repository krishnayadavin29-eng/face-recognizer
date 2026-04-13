"""
augment.py — Data augmentation for face images.
Takes a single RGB face image (numpy array) and returns a list of variants.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def augment_face(rgb_img: np.ndarray, size: int = 160) -> list:
    """
    Return a list of augmented RGB numpy arrays (all resized to `size`×`size`).
    Typically produces 10–12 variants from one input frame.
    """
    pil = Image.fromarray(rgb_img).resize((size, size), Image.LANCZOS)
    variants = []

    # 1. Original
    variants.append(np.array(pil))

    # 2. Horizontal flip
    variants.append(np.array(pil.transpose(Image.FLIP_LEFT_RIGHT)))

    # 3–4. Rotation ±12°
    for angle in (-12, 12):
        variants.append(np.array(pil.rotate(angle, resample=Image.BILINEAR)))

    # 5–6. Brightness (dim / bright)
    for factor in (0.6, 1.4):
        variants.append(np.array(ImageEnhance.Brightness(pil).enhance(factor)))

    # 7. Low contrast
    variants.append(np.array(ImageEnhance.Contrast(pil).enhance(0.6)))

    # 8. Slight blur
    variants.append(np.array(pil.filter(ImageFilter.GaussianBlur(radius=1.5))))

    # 9. Grayscale (converted back to RGB so embedding still works)
    gray = pil.convert("L").convert("RGB")
    variants.append(np.array(gray))

    # 10. Gaussian noise
    arr   = np.array(pil, dtype=np.float32)
    noise = np.random.normal(0, 12, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    variants.append(noisy)

    # 11. Random crop + resize (simulate partial face)
    w, h    = pil.size
    margin  = int(w * 0.12)
    x1, y1  = np.random.randint(0, margin), np.random.randint(0, margin)
    x2, y2  = w - np.random.randint(0, margin), h - np.random.randint(0, margin)
    cropped = pil.crop((x1, y1, x2, y2)).resize((size, size), Image.LANCZOS)
    variants.append(np.array(cropped))

    return variants
