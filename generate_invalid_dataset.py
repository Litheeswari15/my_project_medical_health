"""
generate_invalid_dataset.py
===========================
Generates synthetic "invalid" images for training the rejection class in both models.

Since we cannot download external datasets, this script generates diverse
synthetic images that the models should learn to REJECT:

For retinal model invalid set:
  - Skin-tone uniform images (dermoscopy-like)
  - Bright white/grey uniform images
  - High-contrast random textures
  - Gradient images (no circular structure)
  - Blue/green dominant images
  - Striped/gridded patterns
  - Blurry coloured blobs

For skin model invalid set:
  - Dark circular images on black (fundus-like)
  - Very dark images overall
  - High-frequency noise images
  - Grey uniform images
  - Artificial geometric patterns

Additionally uses HAM10000 metadata CSV to cross-generate:
  - Skin images are used as invalid for the retinal model (if actual images available)
  - Retinal domain images are used as invalid for the skin model

Run this ONCE before training:
  python generate_invalid_dataset.py

Outputs:
  dataset/invalid_for_retinal/   (put here: non-retinal images)
  dataset/invalid_for_skin/      (put here: non-skin images)
"""

import numpy as np
import cv2
import os
import random
import math

random.seed(42)
np.random.seed(42)

IMG_SIZE = 224
IMAGES_PER_TYPE = 150   # synthetic images per pattern type
                         # Increase for better rejection training (e.g. 300)

OUT_RETINAL_INVALID = "dataset/invalid_for_retinal"
OUT_SKIN_INVALID    = "dataset/invalid_for_skin"

os.makedirs(OUT_RETINAL_INVALID, exist_ok=True)
os.makedirs(OUT_SKIN_INVALID, exist_ok=True)


# ─────────────────────────────────────────────
# GENERATOR FUNCTIONS
# ─────────────────────────────────────────────

def save(img, folder, prefix, idx):
    """Clamp, convert to uint8, save."""
    img = np.clip(img, 0, 255).astype(np.uint8)
    path = os.path.join(folder, f"{prefix}_{idx:04d}.jpg")
    cv2.imwrite(path, img)


def gen_uniform_skin_tone(n, folder, prefix):
    """Solid warm skin-tone images — look like dermoscopy backgrounds."""
    for i in range(n):
        # Random skin tones: pinkish-tan range
        b = random.randint(100, 160)
        g = random.randint(120, 175)
        r = random.randint(150, 220)
        # Add slight gaussian noise
        img = np.full((IMG_SIZE, IMG_SIZE, 3), [b, g, r], dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 18), img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_bright_uniform(n, folder, prefix):
    """Very bright nearly-white images."""
    for i in range(n):
        base = random.randint(200, 255)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), base, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 12), img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_gradient(n, folder, prefix):
    """Smooth colour gradients — no circular structure."""
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        c1 = np.array([random.randint(50,255), random.randint(50,255), random.randint(50,255)], dtype=np.float32)
        c2 = np.array([random.randint(50,255), random.randint(50,255), random.randint(50,255)], dtype=np.float32)
        for y in range(IMG_SIZE):
            t = y / IMG_SIZE
            img[y, :] = (1-t)*c1 + t*c2
        noise = np.random.normal(0, 5, img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_blue_green_dominant(n, folder, prefix):
    """Blue/cyan/green dominant images — retinal is red dominant."""
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        mode = i % 3
        if mode == 0:   # Blue dominant
            img[:,:,0] = random.randint(150, 240)
            img[:,:,1] = random.randint(60, 130)
            img[:,:,2] = random.randint(30, 80)
        elif mode == 1: # Green dominant
            img[:,:,0] = random.randint(40, 100)
            img[:,:,1] = random.randint(150, 230)
            img[:,:,2] = random.randint(40, 100)
        else:           # Cyan
            img[:,:,0] = random.randint(120, 220)
            img[:,:,1] = random.randint(150, 230)
            img[:,:,2] = random.randint(30, 80)
        noise = np.random.normal(0, random.uniform(5, 20), img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_striped_pattern(n, folder, prefix):
    """Horizontal/vertical stripes — clearly not medical."""
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        stripe_w = random.randint(4, 30)
        c1 = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.float32)
        c2 = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.float32)
        if i % 2 == 0:  # horizontal
            for y in range(IMG_SIZE):
                img[y, :] = c1 if (y // stripe_w) % 2 == 0 else c2
        else:           # vertical
            for x in range(IMG_SIZE):
                img[:, x] = c1 if (x // stripe_w) % 2 == 0 else c2
        noise = np.random.normal(0, 5, img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_random_texture(n, folder, prefix):
    """High-frequency noise textures."""
    for i in range(n):
        img = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8).astype(np.float32)
        # Smooth slightly so it's not pure noise
        sigma = random.uniform(0.5, 3.0)
        k = max(3, int(sigma*3)*2+1)
        img = cv2.GaussianBlur(img, (k, k), sigma)
        save(img, folder, prefix, i)


def gen_geometric(n, folder, prefix):
    """Random geometric shapes on plain background."""
    for i in range(n):
        bg = random.randint(200, 255)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), bg, dtype=np.float32)
        num_shapes = random.randint(3, 12)
        for _ in range(num_shapes):
            color = (random.randint(0,200), random.randint(0,200), random.randint(0,200))
            shape = random.randint(0, 2)
            if shape == 0:  # rectangle
                x1, y1 = random.randint(0, IMG_SIZE-20), random.randint(0, IMG_SIZE-20)
                x2, y2 = min(x1+random.randint(10,80), IMG_SIZE-1), min(y1+random.randint(10,80), IMG_SIZE-1)
                cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
            elif shape == 1:  # circle
                cx, cy = random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20)
                r = random.randint(5, 40)
                cv2.circle(img, (cx, cy), r, color, -1)
            else:  # triangle
                pts = np.array([[random.randint(0,IMG_SIZE), random.randint(0,IMG_SIZE)] for _ in range(3)])
                cv2.fillPoly(img, [pts], color)
        save(img, folder, prefix, i)


def gen_checkerboard(n, folder, prefix):
    """Checkerboard patterns."""
    for i in range(n):
        sq = random.randint(8, 40)
        c1 = (random.randint(0,150), random.randint(0,150), random.randint(0,150))
        c2 = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for y in range(IMG_SIZE):
            for x in range(0, IMG_SIZE, sq*2):
                row = (y // sq) % 2
                if row == 0:
                    img[y, x:x+sq] = c1
                    img[y, x+sq:x+sq*2] = c2
                else:
                    img[y, x:x+sq] = c2
                    img[y, x+sq:x+sq*2] = c1
        save(img, folder, prefix, i)


def gen_blurry_blobs(n, folder, prefix):
    """Soft coloured blob images — no medical structure."""
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        bg = random.randint(180, 240)
        img[:] = bg
        num_blobs = random.randint(3, 8)
        for _ in range(num_blobs):
            cx = random.randint(20, IMG_SIZE-20)
            cy = random.randint(20, IMG_SIZE-20)
            r  = random.randint(20, 80)
            color = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.float32)
            Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
            mask = np.clip(1 - dist/r, 0, 1)[:,:,np.newaxis]
            img = img * (1-mask) + color * mask
        save(img, folder, prefix, i)


# ─────────────────────────────────────────────
# INVALID FOR RETINAL MODEL
# Things that are NOT fundus/retinal images
# ─────────────────────────────────────────────

def gen_fundus_like_dark_circular(n, folder, prefix):
    """
    Fake fundus-like image but wrong content inside the circle.
    Teaches retinal model that shape alone isn't enough.
    """
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        cx, cy = IMG_SIZE//2, IMG_SIZE//2
        radius = int(IMG_SIZE * 0.45)

        # Random non-retinal content inside circle
        mode = i % 4
        if mode == 0:   # skin tone fill
            b, g, r = random.randint(100,160), random.randint(120,175), random.randint(150,220)
            cv2.circle(img, (cx,cy), radius, (b,g,r), -1)
        elif mode == 1: # grey/white fill (no vessel structure)
            val = random.randint(150, 230)
            cv2.circle(img, (cx,cy), radius, (val,val,val), -1)
        elif mode == 2: # random colour blobs inside circle
            col = (random.randint(0,200), random.randint(200,255), random.randint(0,200))
            cv2.circle(img, (cx,cy), radius, col, -1)
        else:           # blue fill
            cv2.circle(img, (cx,cy), radius, (200, 100, 50), -1)

        noise = np.random.normal(0, random.uniform(5, 20), img.shape)
        img += noise
        save(img, folder, prefix, i)


# ─────────────────────────────────────────────
# INVALID FOR SKIN MODEL
# Things that are NOT dermoscopy/skin images
# ─────────────────────────────────────────────

def gen_fundus_synthetic(n, folder, prefix):
    """
    Synthetic fundus-like images (dark circular on black).
    Teaches skin model to reject retinal-type images.
    """
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        cx, cy = IMG_SIZE//2 + random.randint(-10,10), IMG_SIZE//2 + random.randint(-10,10)
        radius = int(IMG_SIZE * random.uniform(0.38, 0.48))

        # Orange-red retina-like fill
        r_val = random.randint(140, 200)
        g_val = random.randint(60, 110)
        b_val = random.randint(20, 60)
        cv2.circle(img, (cx,cy), radius, (b_val, g_val, r_val), -1)

        # Add some vessel-like lines
        num_vessels = random.randint(3, 8)
        for _ in range(num_vessels):
            angle = random.uniform(0, 2*math.pi)
            length = random.randint(20, radius-5)
            x1 = int(cx + 5*math.cos(angle))
            y1 = int(cy + 5*math.sin(angle))
            x2 = int(cx + length*math.cos(angle))
            y2 = int(cy + length*math.sin(angle))
            thickness = random.randint(1, 3)
            cv2.line(img, (x1,y1), (x2,y2), (b_val//2, g_val//3, r_val//3), thickness)

        noise = np.random.normal(0, random.uniform(3, 10), img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_grey_uniform(n, folder, prefix):
    """Grey/grayscale images — not skin dermoscopy."""
    for i in range(n):
        val = random.randint(30, 220)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), val, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 15), img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_very_dark(n, folder, prefix):
    """Very dark images — dermoscopy is always well lit."""
    for i in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        max_val = random.randint(10, 50)
        noise = np.random.normal(0, random.uniform(1, max_val//2), img.shape)
        img += noise
        save(img, folder, prefix, i)


def gen_xray_like(n, folder, prefix):
    """X-ray-like images: mostly dark with bright bone-like structures."""
    for i in range(n):
        img = np.full((IMG_SIZE, IMG_SIZE, 3), random.randint(5, 30), dtype=np.float32)
        # Add bright ellipsoidal regions
        num_bones = random.randint(2, 6)
        for _ in range(num_bones):
            cx = random.randint(20, IMG_SIZE-20)
            cy = random.randint(20, IMG_SIZE-20)
            rx = random.randint(15, 60)
            ry = random.randint(10, 50)
            angle = random.randint(0, 180)
            brightness = random.randint(150, 240)
            cv2.ellipse(img, (cx,cy), (rx,ry), angle, 0, 360, (brightness,brightness,brightness), -1)
        # Blur to smooth
        img = cv2.GaussianBlur(img, (7,7), 2)
        noise = np.random.normal(0, random.uniform(3, 10), img.shape)
        img += noise
        save(img, folder, prefix, i)


# ─────────────────────────────────────────────
# CROSS-DOMAIN: Use HAM10000 metadata to copy
# real skin image_ids as hints for naming
# (actual images not available in this env,
#  but names are recorded for future use)
# ─────────────────────────────────────────────

def record_ham_image_ids():
    """Parse HAM10000 CSV and save image IDs per class."""
    import csv
    csv_path = "HAM10000_metadata.csv"
    if not os.path.exists(csv_path):
        print(f"  HAM10000_metadata.csv not found, skipping cross-domain naming")
        return

    class_ids = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            dx = row['dx']
            if dx not in class_ids:
                class_ids[dx] = []
            class_ids[dx].append(row['image_id'])

    # Save a reference file — useful when actual images become available
    import json
    with open("dataset/ham10000_class_ids.json", "w") as f:
        json.dump(class_ids, f, indent=2)
    print(f"  Saved dataset/ham10000_class_ids.json with {sum(len(v) for v in class_ids.values())} image IDs")
    print(f"  Classes: { {k: len(v) for k,v in class_ids.items()} }")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    N = IMAGES_PER_TYPE
    os.makedirs("dataset", exist_ok=True)

    print("=" * 60)
    print("GENERATING INVALID IMAGES FOR RETINAL MODEL")
    print(f"Output: {OUT_RETINAL_INVALID}")
    print("=" * 60)

    print(f"  Generating {N} uniform skin-tone images...")
    gen_uniform_skin_tone(N, OUT_RETINAL_INVALID, "skin_tone")

    print(f"  Generating {N} bright uniform images...")
    gen_bright_uniform(N, OUT_RETINAL_INVALID, "bright")

    print(f"  Generating {N} gradient images...")
    gen_gradient(N, OUT_RETINAL_INVALID, "gradient")

    print(f"  Generating {N} blue/green dominant images...")
    gen_blue_green_dominant(N, OUT_RETINAL_INVALID, "bluegrn")

    print(f"  Generating {N} striped pattern images...")
    gen_striped_pattern(N, OUT_RETINAL_INVALID, "stripe")

    print(f"  Generating {N} geometric shape images...")
    gen_geometric(N, OUT_RETINAL_INVALID, "geom")

    print(f"  Generating {N} blurry blob images...")
    gen_blurry_blobs(N, OUT_RETINAL_INVALID, "blob")

    print(f"  Generating {N} dark-circular (wrong content) images...")
    gen_fundus_like_dark_circular(N, OUT_RETINAL_INVALID, "dark_circ")

    print(f"  Generating {N} checkerboard images...")
    gen_checkerboard(N, OUT_RETINAL_INVALID, "checker")

    print(f"  Generating {N} random texture images...")
    gen_random_texture(N, OUT_RETINAL_INVALID, "texture")

    total_ret = len(os.listdir(OUT_RETINAL_INVALID))
    print(f"\n  ✅ {total_ret} invalid images for retinal model")

    print()
    print("=" * 60)
    print("GENERATING INVALID IMAGES FOR SKIN MODEL")
    print(f"Output: {OUT_SKIN_INVALID}")
    print("=" * 60)

    print(f"  Generating {N} synthetic fundus-like images...")
    gen_fundus_synthetic(N, OUT_SKIN_INVALID, "fundus")

    print(f"  Generating {N} grey uniform images...")
    gen_grey_uniform(N, OUT_SKIN_INVALID, "grey")

    print(f"  Generating {N} very dark images...")
    gen_very_dark(N, OUT_SKIN_INVALID, "dark")

    print(f"  Generating {N} X-ray-like images...")
    gen_xray_like(N, OUT_SKIN_INVALID, "xray")

    print(f"  Generating {N} geometric images...")
    gen_geometric(N, OUT_SKIN_INVALID, "geom")

    print(f"  Generating {N} random texture images...")
    gen_random_texture(N, OUT_SKIN_INVALID, "texture")

    print(f"  Generating {N} blue/green dominant images...")
    gen_blue_green_dominant(N, OUT_SKIN_INVALID, "bluegrn")

    print(f"  Generating {N} gradient images...")
    gen_gradient(N, OUT_SKIN_INVALID, "gradient")

    total_skin = len(os.listdir(OUT_SKIN_INVALID))
    print(f"\n  ✅ {total_skin} invalid images for skin model")

    print()
    print("=" * 60)
    print("RECORDING HAM10000 IMAGE IDs")
    print("=" * 60)
    record_ham_image_ids()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  dataset/invalid_for_retinal/ : {total_ret} images")
    print(f"  dataset/invalid_for_skin/    : {total_skin} images")
    print()
    print("  Next steps:")
    print("  1. Run: python train.py")
    print("  2. Run: python train_skin_model.py")
    print("  Both scripts will automatically use these generated images.")
    print()
    print("  OPTIONAL (for stronger rejection):")
    print("  - Add real non-retinal images to dataset/invalid_for_retinal/")
    print("  - Add real non-skin images to dataset/invalid_for_skin/")
    print("  - E.g. copy HAM10000 images into dataset/invalid_for_retinal/")
    print("  - E.g. copy APTOS retinal images into dataset/invalid_for_skin/")


if __name__ == "__main__":
    main()
