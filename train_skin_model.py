import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import os
import glob
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = 224
BATCH       = 32
EPOCHS      = 15
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
INVALID_DIR = "dataset/invalid_for_skin"

SKIN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "invalid"]
NUM_CLASSES  = len(SKIN_CLASSES)
INVALID_IDX  = SKIN_CLASSES.index("invalid")

CLASS_FULL_NAMES = {
    "akiec":   "Actinic Keratosis",
    "bcc":     "Basal Cell Carcinoma",
    "bkl":     "Benign Keratosis",
    "df":      "Dermatofibroma",
    "mel":     "Melanoma",
    "nv":      "Melanocytic Nevi",
    "vasc":    "Vascular Lesion",
    "invalid": "Invalid / Not a skin image",
}

# ─────────────────────────────────────────────
# COLLECT FILE PATHS ONLY — no pixel data in RAM
# ─────────────────────────────────────────────
def collect_file_paths():
    paths, labels = [], []

    for cls_idx, cls_name in enumerate(SKIN_CLASSES[:-1]):
        for split_dir in [TRAIN_DIR, TEST_DIR]:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            files = (glob.glob(os.path.join(cls_dir, "*.jpg"))  +
                     glob.glob(os.path.join(cls_dir, "*.jpeg")) +
                     glob.glob(os.path.join(cls_dir, "*.png")))
            paths.extend(files)
            labels.extend([cls_idx] * len(files))

    if os.path.exists(INVALID_DIR):
        inv_files = (glob.glob(os.path.join(INVALID_DIR, "*.jpg"))  +
                     glob.glob(os.path.join(INVALID_DIR, "*.jpeg")) +
                     glob.glob(os.path.join(INVALID_DIR, "*.png")))
        paths.extend(inv_files)
        labels.extend([INVALID_IDX] * len(inv_files))
    else:
        print(f"WARNING: {INVALID_DIR} not found — run generate_invalid_dataset.py first")

    return np.array(paths), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────
# DISK-BASED GENERATOR
# Reads one batch from disk per step — never holds full dataset in RAM
# ─────────────────────────────────────────────
class SkinDataGenerator(Sequence):
    def __init__(self, paths, labels, batch_size, num_classes,
                 augment=False, shuffle=True):
        self.paths      = paths
        self.labels     = labels
        self.batch_size = batch_size
        self.num_classes= num_classes
        self.augment    = augment
        self.shuffle    = shuffle
        self.indexes    = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return max(1, len(self.paths) // self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]
        X = np.zeros((len(batch_idx), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        y = np.zeros((len(batch_idx), self.num_classes), dtype=np.float32)

        for i, file_idx in enumerate(batch_idx):
            img = self._load(self.paths[file_idx])
            if self.augment:
                img = self._augment(img)
            X[i] = img
            y[i] = tf.keras.utils.to_categorical(self.labels[file_idx], self.num_classes)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0

    def _augment(self, img):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        if random.random() > 0.5:
            scale = random.uniform(0.85, 1.0)
            new_h = int(IMG_SIZE * scale)
            new_w = int(IMG_SIZE * scale)
            top  = random.randint(0, IMG_SIZE - new_h)
            left = random.randint(0, IMG_SIZE - new_w)
            img  = img[top:top+new_h, left:left+new_w]
            img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32)


# ─────────────────────────────────────────────
# COLLECT PATHS
# ─────────────────────────────────────────────
print("=" * 60)
print("COLLECTING FILE PATHS (no RAM loading)")
print("=" * 60)

all_paths, all_labels = collect_file_paths()

print(f"\nTotal files: {len(all_paths)}")
for i, name in enumerate(SKIN_CLASSES):
    print(f"  {name:8s} (idx={i}): {int(np.sum(all_labels == i))}")

# ─────────────────────────────────────────────
# TRAIN / VAL SPLIT (paths only)
# ─────────────────────────────────────────────
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.2, stratify=all_labels, random_state=42
)

print(f"\nTrain: {len(train_paths)}  Val: {len(val_paths)}")

# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
cw_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)
class_weight_dict = {i: float(w) for i, w in enumerate(cw_arr)}
class_weight_dict[INVALID_IDX] *= 2.0
print("\nClass weights:")
for k, v in class_weight_dict.items():
    print(f"  {SKIN_CLASSES[k]:8s}: {v:.2f}")

# ─────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────
train_gen = SkinDataGenerator(train_paths, train_labels, BATCH, NUM_CLASSES,
                               augment=True, shuffle=True)
val_gen   = SkinDataGenerator(val_paths,   val_labels,   BATCH, NUM_CLASSES,
                               augment=False, shuffle=False)

print(f"\nTrain steps/epoch: {len(train_gen)}")
print(f"Val steps/epoch  : {len(val_gen)}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
print("\nBuilding model...")

base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
print(f"\nModel: {NUM_CLASSES} classes — index {INVALID_IDX} = 'invalid' (rejection)")

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max',
        patience=4, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', mode='min',
        factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        "skin_model_checkpoint.keras",
        monitor='val_accuracy', mode='max',
        save_best_only=True, verbose=1),
]

# ─────────────────────────────────────────────
# PHASE 1: Frozen base
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Frozen MobileNetV2 base")
print("="*60)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# PHASE 2: Fine-tune top 30 layers
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: Fine-tuning top 30 MobileNetV2 layers")
print("="*60)

for layer in base.layers[-30:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# EVALUATE — batch-by-batch to avoid OOM
# ─────────────────────────────────────────────
print("\nEvaluating rejection class on validation set...")

eval_gen = SkinDataGenerator(val_paths, val_labels, 64, NUM_CLASSES,
                              augment=False, shuffle=False)
all_preds, all_true = [], []
for i in range(len(eval_gen)):
    xb, yb = eval_gen[i]
    pb = model.predict(xb, verbose=0)
    all_preds.extend(np.argmax(pb, axis=1))
    all_true.extend(np.argmax(yb, axis=1))

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

print(f"  Overall accuracy : {np.mean(all_preds == all_true):.1%}")

inv_mask = (all_true == INVALID_IDX)
if inv_mask.sum() > 0:
    ic = np.sum(all_preds[inv_mask] == INVALID_IDX)
    it = inv_mask.sum()
    print(f"  Rejection accuracy : {ic}/{it} = {ic/it:.1%}")
    fp = int(np.sum((all_preds != INVALID_IDX) & (all_true == INVALID_IDX)))
    print(f"  False positives (invalid accepted as skin): {fp}")

valid_mask = (all_true != INVALID_IDX)
if valid_mask.sum() > 0:
    print(f"  Skin accuracy (valid only): {np.mean(all_preds[valid_mask]==all_true[valid_mask]):.1%}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
model.save("skin_model.keras")
with open("skin_classes.json", "w") as f:
    json.dump(SKIN_CLASSES, f, indent=2)
with open("skin_class_names.json", "w") as f:
    json.dump(CLASS_FULL_NAMES, f, indent=2)

print("\nskin_model.keras saved!")
print("skin_classes.json saved!")
print("skin_class_names.json saved!")
print(f"Classes: {SKIN_CLASSES}")