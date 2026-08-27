import pandas as pd
import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 15

# Path to APTOS 2019 dataset (change to your actual path)
DATA_DIR = "E:/MY_PROJECT/dataset/train_images/train_images"
CSV_PATH = "E:/MY_PROJECT/dataset/train.csv"

# Generated invalid images (run generate_invalid_dataset.py first)
INVALID_DIR = "dataset/invalid_for_retinal"

# ─────────────────────────────────────────────
# IMAGE LOADER
# ─────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def load_by_id(image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(DATA_DIR, image_id + ext)
        if os.path.exists(path):
            img = load_image(path)
            if img is not None:
                return img
    return None

# ─────────────────────────────────────────────
# LOAD RETINAL DATASET (APTOS 2019)
# ─────────────────────────────────────────────
print("=" * 60)
print("LOADING RETINAL DATASET (APTOS 2019)")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"CSV entries: {len(df)}")
print(f"DR distribution:\n{df['diagnosis'].value_counts().sort_index()}")

images      = []
labels_dr   = []
labels_heart= []
labels_hyper= []
labels_valid= []   # 1 = valid retinal, 0 = invalid

for idx, row in df.iterrows():
    img = load_by_id(row['id_code'])
    if img is None:
        continue

    dr    = int(row['diagnosis'])
    heart = 1 if dr >= 2 else 0
    hyper = 1 if dr >= 3 else 0

    images.append(img)
    labels_dr.append(dr)
    labels_heart.append(heart)
    labels_hyper.append(hyper)
    labels_valid.append(1)   # valid retinal

    if (idx+1) % 500 == 0:
        print(f"  Loaded {idx+1}/{len(df)} retinal images...")

n_retinal = len(images)
print(f"✅ Loaded {n_retinal} retinal images")

# ─────────────────────────────────────────────
# LOAD INVALID IMAGES
# ─────────────────────────────────────────────
print(f"\nLOADING INVALID IMAGES from {INVALID_DIR}")

invalid_count = 0
if os.path.exists(INVALID_DIR):
    all_invalid = glob.glob(os.path.join(INVALID_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(INVALID_DIR, "*.png"))

    for path in all_invalid:
        img = load_image(path)
        if img is None:
            continue
        images.append(img)
        labels_dr.append(0)     # placeholder
        labels_heart.append(0)  # placeholder
        labels_hyper.append(0)  # placeholder
        labels_valid.append(0)  # INVALID
        invalid_count += 1

    print(f"✅ Loaded {invalid_count} invalid images")
else:
    print(f"⚠️  {INVALID_DIR} not found.")
    print("    Run: python generate_invalid_dataset.py")
    print("    Continuing without rejection training (model won't reject wrong images)")

# ─────────────────────────────────────────────
# CONVERT TO ARRAYS
# ─────────────────────────────────────────────
images       = np.array(images,       dtype=np.float32)
labels_dr    = np.array(labels_dr,    dtype=np.int32)
labels_heart = np.array(labels_heart, dtype=np.float32)
labels_hyper = np.array(labels_hyper, dtype=np.float32)
labels_valid = np.array(labels_valid, dtype=np.float32)

print(f"\nDataset: {len(images)} total images")
print(f"  Valid retinal : {int(labels_valid.sum())}")
print(f"  Invalid       : {int((1-labels_valid).sum())}")

# ─────────────────────────────────────────────
# TRAIN / VAL SPLIT
# ─────────────────────────────────────────────
(X_train, X_val,
 y_dr_tr,  y_dr_val,
 y_h_tr,   y_h_val,
 y_hp_tr,  y_hp_val,
 y_v_tr,   y_v_val) = train_test_split(
    images, labels_dr, labels_heart, labels_hyper, labels_valid,
    test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
print("\nBuilding model...")

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# 4 outputs: validity gate + 3 disease predictions
valid_output = layers.Dense(1, activation='sigmoid', name='valid_output')(x)
dr_output    = layers.Dense(5, activation='softmax', name='dr_output')(x)
heart_output = layers.Dense(1, activation='sigmoid', name='heart_output')(x)
hyper_output = layers.Dense(1, activation='sigmoid', name='hyper_output')(x)

model = models.Model(
    inputs=base.input,
    outputs=[valid_output, dr_output, heart_output, hyper_output]
)

# ─────────────────────────────────────────────
# COMPILE
# ─────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'valid_output': 'binary_crossentropy',
        'dr_output':    'sparse_categorical_crossentropy',
        'heart_output': 'binary_crossentropy',
        'hyper_output': 'binary_crossentropy',
    },
    loss_weights={
        'valid_output': 2.0,   # Validity is highest priority
        'dr_output':    1.0,
        'heart_output': 0.5,
        'hyper_output': 0.5,
    },
    metrics={
        'valid_output': ['accuracy'],
        'dr_output':    ['accuracy'],
        'heart_output': ['accuracy'],
        'hyper_output': ['accuracy'],
    }
)

model.summary()

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_valid_output_accuracy',
        mode='max',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "model_checkpoint.keras",
        monitor='val_valid_output_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# TRAIN — PHASE 1: Frozen base
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Training with frozen MobileNetV2 base")
print("="*60)

history = model.fit(
    X_train,
    {
        'valid_output': y_v_tr,
        'dr_output':    y_dr_tr,
        'heart_output': y_h_tr,
        'hyper_output': y_hp_tr,
    },
    validation_data=(
        X_val,
        {
            'valid_output': y_v_val,
            'dr_output':    y_dr_val,
            'heart_output': y_h_val,
            'hyper_output': y_hp_val,
        }
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# TRAIN — PHASE 2: Fine-tune top layers of base
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: Fine-tuning top 30 layers of MobileNetV2")
print("="*60)

# Unfreeze top 30 layers
for layer in base.layers[-30:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'valid_output': 'binary_crossentropy',
        'dr_output':    'sparse_categorical_crossentropy',
        'heart_output': 'binary_crossentropy',
        'hyper_output': 'binary_crossentropy',
    },
    loss_weights={
        'valid_output': 2.0,
        'dr_output':    1.0,
        'heart_output': 0.5,
        'hyper_output': 0.5,
    },
    metrics={
        'valid_output': ['accuracy'],
        'dr_output':    ['accuracy'],
        'heart_output': ['accuracy'],
        'hyper_output': ['accuracy'],
    }
)

model.fit(
    X_train,
    {
        'valid_output': y_v_tr,
        'dr_output':    y_dr_tr,
        'heart_output': y_h_tr,
        'hyper_output': y_hp_tr,
    },
    validation_data=(
        X_val,
        {
            'valid_output': y_v_val,
            'dr_output':    y_dr_val,
            'heart_output': y_h_val,
            'hyper_output': y_hp_val,
        }
    ),
    epochs=8,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# EVALUATE VALIDITY GATE
# ─────────────────────────────────────────────
print("\n📊 Evaluating validity gate...")
preds = model.predict(X_val, verbose=0)
valid_preds  = (preds[0].flatten() > 0.65).astype(int)
valid_actual = y_v_val.astype(int)

acc = np.mean(valid_preds == valid_actual)
fp  = int(np.sum((valid_preds==1) & (valid_actual==0)))
fn  = int(np.sum((valid_preds==0) & (valid_actual==1)))
tp  = int(np.sum((valid_preds==1) & (valid_actual==1)))
tn  = int(np.sum((valid_preds==0) & (valid_actual==0)))

print(f"  Validity gate accuracy : {acc:.1%}")
print(f"  True  positives (correct accept) : {tp}")
print(f"  True  negatives (correct reject) : {tn}")
print(f"  False positives (wrong img accepted): {fp}  ← lower is better")
print(f"  False negatives (valid img rejected): {fn}")

dr_preds  = np.argmax(preds[1], axis=1)
dr_actual = y_dr_val
# Only evaluate DR on valid images
valid_mask = valid_actual == 1
if valid_mask.sum() > 0:
    dr_acc = np.mean(dr_preds[valid_mask] == dr_actual[valid_mask])
    print(f"  DR classification accuracy (valid only): {dr_acc:.1%}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
model.save("model.keras")
print("\n✅ model.keras saved!")
print("   Outputs: [valid_output, dr_output, heart_output, hyper_output]")