"""
test_skin_model.py
==================
Tests the skin model on a single image using the full validation pipeline.

Usage:
  python test_skin_model.py <image_path>
  python test_skin_model.py test.jpg

The script will:
  1. Run heuristic pre-checks (fast, no model)
  2. Load the model and run prediction
  3. Print whether the image was accepted or rejected
  4. If accepted, show the diagnosis + confidence
"""

import sys
import os
import tensorflow as tf
from predict import preprocess, predict_skin

if len(sys.argv) < 2:
    img_path = "test.jpg"
    print(f"No image path given, using default: {img_path}")
else:
    img_path = sys.argv[1]

if not os.path.exists(img_path):
    print(f"❌ File not found: {img_path}")
    sys.exit(1)

print(f"Loading image: {img_path}")
img_raw, img_batch = preprocess(img_path)

print("Loading skin_model.keras...")
model = tf.keras.models.load_model("skin_model.keras")

print("Running prediction with validation...\n")
result = predict_skin(img_raw, img_batch, model)

if not result["valid"]:
    print("❌ IMAGE REJECTED")
    print(f"   Reason: {result['reason']}")
else:
    print("✅ IMAGE ACCEPTED")
    print(f"   Diagnosis  : {result['diagnosis']}")
    print(f"   Class code : {result['label']}")
    print(f"   Confidence : {result['confidence']:.1%}")