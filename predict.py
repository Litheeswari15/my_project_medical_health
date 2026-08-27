"""
predict.py
==========
Unified prediction + validation pipeline for SmartVision AI.

Provides:
  - Image heuristic pre-checks (fast, before model inference)
  - predict_retinal() — validates + predicts DR/Heart/Hypertension
  - predict_skin()    — validates + predicts skin lesion class
  - preprocess_bytes() — for Streamlit uploaded files
  - preprocess()      — for file path (CLI use)

Used by app.py (Streamlit) and can also be run standalone:
  python predict.py retinal <image_path>
  python predict.py skin    <image_path>
"""

import numpy as np
import cv2
import os
import json
import sys

IMG_SIZE = 224

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
# How confident the retinal model's validity gate must be.
# 0.65 = model must be 65%+ sure it's a retinal image.
RETINAL_VALIDITY_THRESHOLD = 0.65

# Minimum softmax confidence for skin classification.
# Below this, reject even if model picked a valid class.
SKIN_CONFIDENCE_THRESHOLD = 0.40

# ─────────────────────────────────────────────
# SKIN CLASS LABELS
# ─────────────────────────────────────────────
def _load_json(path, fallback):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return fallback

SKIN_CLASSES = _load_json(
    "skin_classes.json",
    ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "invalid"]
)

SKIN_CLASS_NAMES = _load_json(
    "skin_class_names.json",
    {
        "akiec":   "Actinic Keratosis",
        "bcc":     "Basal Cell Carcinoma",
        "bkl":     "Benign Keratosis",
        "df":      "Dermatofibroma",
        "mel":     "Melanoma",
        "nv":      "Melanocytic Nevi",
        "vasc":    "Vascular Lesion",
        "invalid": "Invalid / Not a skin image",
    }
)

# ─────────────────────────────────────────────
# HEURISTIC PRE-CHECKS
# These run BEFORE the model — fast rejection of obviously wrong images.
# ─────────────────────────────────────────────

def check_retinal_heuristics(img_bgr):
    """
    Retinal fundus images have a distinctive signature:
      • Large circular bright region (the fundus) on a BLACK background
      • Red/orange dominant colour channel (blood vessels + retinal pigment)
      • Always roughly square (1:1 aspect ratio)
      • No extremely uniform or bright backgrounds

    Returns: (passes: bool, reason: str)
    """
    h, w   = img_bgr.shape[:2]
    pixels = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Aspect ratio check — fundus cameras produce ~square images
    aspect = w / h
    if aspect < 0.4 or aspect > 2.5:
        return False, (
            f"Unusual aspect ratio ({aspect:.2f}) — "
            "retinal fundus images are always roughly square (1:1)."
        )

    # 2. Dark border check — fundus images have significant black border
    dark_pixels = int(np.sum(gray < 20))
    dark_ratio  = dark_pixels / pixels

    # 3. Overall brightness
    mean_brightness = float(np.mean(gray))

    # Very bright uniform image → definitely not fundus
    if mean_brightness > 200 and dark_ratio < 0.02:
        return False, (
            f"Image is too bright (mean={mean_brightness:.0f}) with no dark border — "
            "retinal fundus images always have a circular crop on black."
        )

    # 4. Colour channel analysis
    b_ch = img_bgr[:,:,0].astype(np.float32)
    g_ch = img_bgr[:,:,1].astype(np.float32)
    r_ch = img_bgr[:,:,2].astype(np.float32)
    mean_b, mean_g, mean_r = float(np.mean(b_ch)), float(np.mean(g_ch)), float(np.mean(r_ch))

    # Blue dominant with no dark border → not a fundus image
    if mean_b > mean_r * 1.6 and dark_ratio < 0.10:
        return False, (
            f"Blue-dominant image (B={mean_b:.0f} > R={mean_r:.0f}) — "
            "retinal images are red/orange dominant."
        )

    # Strong green dominance → not retinal
    if mean_g > mean_r * 1.5 and dark_ratio < 0.10:
        return False, (
            f"Green-dominant image (G={mean_g:.0f} > R={mean_r:.0f}) — "
            "retinal images are red/orange dominant."
        )

    # 5. Bright region check — fundus must have a circular bright zone
    _, bright_mask  = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    bright_ratio    = float(np.sum(bright_mask > 0)) / pixels

    if bright_ratio < 0.25:
        return False, (
            f"Bright region too small ({bright_ratio:.1%}) — "
            "the fundus circular region should occupy at least 25% of the image."
        )

    # 6. Skin-tone check — warm tan/pink image with no dark border = dermoscopy
    # Skin tones: R > G > B, with R in 130-230 range
    if (mean_r > 130 and mean_r < 230 and
        mean_r > mean_g > mean_b and
        dark_ratio < 0.05 and
        mean_brightness > 100):
        return False, (
            "Image appears to be a skin-tone photograph — "
            "please upload a fundus/retinal photograph instead."
        )

    return True, "Passes retinal heuristics"


def check_skin_heuristics(img_bgr):
    """
    Dermoscopy skin images have a distinctive signature:
      • Warm skin-tone colours (pink, tan, brown)
      • No large dark border (unlike fundus)
      • Well-lit (not very dark overall)
      • Roughly square (dermoscopes produce square images)

    Returns: (passes: bool, reason: str)
    """
    h, w   = img_bgr.shape[:2]
    pixels = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Aspect ratio
    aspect = w / h
    if aspect < 0.4 or aspect > 3.0:
        return False, (
            f"Unusual aspect ratio ({aspect:.2f}) — "
            "dermoscopy images are always roughly square."
        )

    # 2. Dark border check — fundus images have large black border, skin don't
    dark_pixels = int(np.sum(gray < 20))
    dark_ratio  = dark_pixels / pixels

    if dark_ratio > 0.25:
        return False, (
            f"Large dark border detected ({dark_ratio:.1%} dark pixels) — "
            "this looks like a retinal fundus image, not a skin dermoscopy image."
        )

    # 3. Overall darkness — dermoscopy images are always well-lit
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 35:
        return False, (
            f"Image too dark (mean brightness={mean_brightness:.0f}) — "
            "dermoscopy images are always well-lit."
        )

    # 4. Check for skin-tone hues (HSV)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0].astype(np.float32)
    sat = hsv[:,:,1].astype(np.float32)
    val = hsv[:,:,2].astype(np.float32)

    # Skin hue in OpenCV HSV (0-180): roughly 0-20 and 160-180
    skin_hue_mask = ((hue <= 22) | (hue >= 158))
    skin_hue_ratio = float(np.sum(skin_hue_mask)) / pixels
    mean_sat = float(np.mean(sat))
    mean_val = float(np.mean(val))

    # Very blue/green dominant with no skin tones
    b_ch = img_bgr[:,:,0].astype(np.float32)
    g_ch = img_bgr[:,:,1].astype(np.float32)
    r_ch = img_bgr[:,:,2].astype(np.float32)
    mean_b = float(np.mean(b_ch))
    mean_g = float(np.mean(g_ch))
    mean_r = float(np.mean(r_ch))

    if mean_b > mean_r * 1.8 and skin_hue_ratio < 0.2:
        return False, (
            f"Blue-dominant image with no skin tones — "
            "please upload a dermoscopy or skin photograph."
        )

    # Grey/grayscale image — dermoscopy is always colour
    channel_variance = np.var([mean_b, mean_g, mean_r])
    if channel_variance < 25 and mean_brightness > 50:
        return False, (
            "Grayscale or near-grayscale image — "
            "dermoscopy images are always colour photographs."
        )

    return True, "Passes skin heuristics"


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(img_path):
    """Load from file path. Returns (img_raw_bgr, img_batch) or raises."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_raw     = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm    = img_raw / 255.0
    img_batch   = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return img_raw, img_batch


def preprocess_bytes(image_bytes):
    """
    Load from raw bytes (Streamlit uploaded file).
    Returns (img_raw_bgr, img_batch) or (None, None) on failure.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img_raw   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm  = img_raw / 255.0
    img_batch = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return img_raw, img_batch


# ─────────────────────────────────────────────
# PREDICT RETINAL
# ─────────────────────────────────────────────

def predict_retinal(img_raw, img_batch, model):
    """
    Validates and predicts from a retinal image.

    Three-layer defence:
      1. Heuristic pre-check (fast, no model)
      2. Model validity gate (valid_output sigmoid)
      3. Only runs disease prediction if both pass

    Args:
      img_raw   : np.ndarray (224,224,3) BGR uint8/float
      img_batch : np.ndarray (1,224,224,3) float32 normalised
      model     : loaded keras model

    Returns dict:
      valid=True  → {dr_stage, dr_label, heart, hyper, confidence}
      valid=False → {reason}
    """
    # ── Layer 1: Heuristics ──
    heuristic_ok, heuristic_reason = check_retinal_heuristics(img_raw)
    if not heuristic_ok:
        return {
            "valid":      False,
            "reason":     heuristic_reason,
            "dr_stage":   None, "dr_label": None,
            "heart":      None, "hyper":    None,
            "confidence": 0.0
        }

    # ── Layer 2: Model prediction ──
    outputs = model.predict(img_batch, verbose=0)

    # Support both old model (3 outputs) and new model (4 outputs)
    if len(outputs) == 4:
        valid_score = float(outputs[0][0][0])
        dr_probs    = outputs[1][0]
        heart_score = float(outputs[2][0][0])
        hyper_score = float(outputs[3][0][0])

        # ── Layer 2: Validity gate ──
        if valid_score < RETINAL_VALIDITY_THRESHOLD:
            return {
                "valid":  False,
                "reason": (
                    f"The model is not confident this is a retinal image "
                    f"(validity score: {valid_score:.1%}, threshold: {RETINAL_VALIDITY_THRESHOLD:.0%}). "
                    f"Please upload a fundus/retinal photograph — a circular eye image "
                    f"with orange/red tones on a dark background."
                ),
                "dr_stage":   None, "dr_label": None,
                "heart":      None, "hyper":    None,
                "confidence": valid_score
            }

    elif len(outputs) == 3:
        # Backwards compatibility with old 3-output model (no validity gate)
        dr_probs    = outputs[0][0]
        heart_score = float(outputs[1][0][0])
        hyper_score = float(outputs[2][0][0])
        valid_score = 1.0
    else:
        return {
            "valid":  False,
            "reason": f"Unexpected model output count: {len(outputs)}",
            "confidence": 0.0
        }

    dr_labels_map = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    dr_stage      = int(np.argmax(dr_probs))
    dr_conf       = float(np.max(dr_probs))
    heart_pred    = "YES" if heart_score > 0.5 else "NO"
    hyper_pred    = "YES" if hyper_score > 0.5 else "NO"

    return {
        "valid":      True,
        "reason":     "Valid retinal image",
        "dr_stage":   dr_stage,
        "dr_label":   dr_labels_map[dr_stage],
        "heart":      heart_pred,
        "hyper":      hyper_pred,
        "confidence": dr_conf,
        "valid_score": valid_score if len(outputs) == 4 else None,
    }


# ─────────────────────────────────────────────
# PREDICT SKIN
# ─────────────────────────────────────────────

def predict_skin(img_raw, img_batch, model):
    """
    Validates and predicts skin lesion class.

    Three-layer defence:
      1. Heuristic pre-check
      2. Model checks for 'invalid' class (index 7)
      3. Confidence threshold check

    Returns dict:
      valid=True  → {diagnosis, label, confidence}
      valid=False → {reason}
    """
    # ── Layer 1: Heuristics ──
    heuristic_ok, heuristic_reason = check_skin_heuristics(img_raw)
    if not heuristic_ok:
        return {
            "valid":      False,
            "reason":     heuristic_reason,
            "diagnosis":  None, "label":     None,
            "confidence": 0.0
        }

    # ── Layer 2: Model prediction ──
    pred       = model.predict(img_batch, verbose=0)[0]
    class_idx  = int(np.argmax(pred))
    confidence = float(np.max(pred))
    label      = SKIN_CLASSES[class_idx] if class_idx < len(SKIN_CLASSES) else "unknown"

    # ── Layer 2: Model says "invalid" ──
    if label == "invalid":
        return {
            "valid":  False,
            "reason": (
                f"The model classified this image as non-skin/invalid "
                f"(confidence: {confidence:.1%}). "
                f"Please upload a dermoscopy or skin lesion photograph — "
                f"a close-up image of a skin lesion with warm skin-tone colours."
            ),
            "diagnosis":  None, "label":     None,
            "confidence": confidence
        }

    # ── Layer 3: Confidence threshold ──
    if confidence < SKIN_CONFIDENCE_THRESHOLD:
        return {
            "valid":  False,
            "reason": (
                f"The model is not confident enough ({confidence:.1%}, "
                f"minimum required: {SKIN_CONFIDENCE_THRESHOLD:.0%}). "
                f"The image may not be a dermoscopy/skin photograph, "
                f"or the lesion may be unclear."
            ),
            "diagnosis":  None, "label":     None,
            "confidence": confidence
        }

    full_name = SKIN_CLASS_NAMES.get(label, label)

    return {
        "valid":      True,
        "reason":     "Valid skin image",
        "diagnosis":  full_name,
        "label":      label,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────
# STANDALONE CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tensorflow as tf

    if len(sys.argv) < 3:
        print("Usage: python predict.py <retinal|skin> <image_path>")
        print()
        print("Examples:")
        print("  python predict.py retinal fundus_image.jpg")
        print("  python predict.py skin    dermoscopy.jpg")
        sys.exit(1)

    mode     = sys.argv[1].lower()
    img_path = sys.argv[2]

    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)

    print(f"Loading image: {img_path}")
    img_raw, img_batch = preprocess(img_path)

    if mode == "retinal":
        print("Loading retinal model (model.keras)...")
        model  = tf.keras.models.load_model("model.keras")
        result = predict_retinal(img_raw, img_batch, model)

        print()
        if not result["valid"]:
            print("❌ IMAGE REJECTED")
            print(f"   Reason: {result['reason']}")
        else:
            print("✅ IMAGE ACCEPTED — Retinal Analysis:")
            print(f"   Diabetic Retinopathy : {result['dr_label']} (stage {result['dr_stage']})")
            print(f"   Heart Disease Risk   : {result['heart']}")
            print(f"   Hypertension Risk    : {result['hyper']}")
            print(f"   DR Confidence        : {result['confidence']:.1%}")
            if result.get('valid_score') is not None:
                print(f"   Validity Score       : {result['valid_score']:.1%}")

    elif mode == "skin":
        print("Loading skin model (skin_model.keras)...")
        model  = tf.keras.models.load_model("skin_model.keras")
        result = predict_skin(img_raw, img_batch, model)

        print()
        if not result["valid"]:
            print("❌ IMAGE REJECTED")
            print(f"   Reason: {result['reason']}")
        else:
            print("✅ IMAGE ACCEPTED — Skin Analysis:")
            print(f"   Diagnosis  : {result['diagnosis']}")
            print(f"   Class code : {result['label']}")
            print(f"   Confidence : {result['confidence']:.1%}")

    else:
        print(f"❌ Unknown mode '{mode}'. Use 'retinal' or 'skin'.")
        sys.exit(1)