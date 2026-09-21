"""
Enhanced Sample Data Generator script.
Generates sample face images with realistic shapes and high contrast features
that trigger OpenCV Haar cascades and face detection models reliably.
"""
import os
import cv2
import numpy as np
import config


def draw_realistic_face(name_seed, variant=0):
    """
    Draw a clean, well-proportioned synthetic face with strong contrast
    that triggers face detection cascades cleanly.
    """
    width, height = 400, 400
    img = np.full((height, width, 3), (235, 235, 235), dtype=np.uint8)

    seed_val = sum(ord(c) for c in name_seed) + variant * 29
    np.random.seed(seed_val)

    # Center coordinates
    cx, cy = 200, 200
    head_w, head_h = 130, 170

    # Face background contour / skin tone (BGR)
    skin_tones = [
        (190, 210, 240),
        (160, 185, 225),
        (130, 160, 210),
        (200, 220, 245),
    ]
    skin_color = skin_tones[seed_val % len(skin_tones)]

    # Hair back
    hair_colors = [(30, 30, 30), (20, 40, 70), (40, 70, 100), (60, 60, 60)]
    hair_color = hair_colors[seed_val % len(hair_colors)]
    cv2.ellipse(img, (cx, cy - 25), (head_w + 15, head_h + 15), 0, 180, 360, hair_color, -1)

    # Face Oval
    cv2.ellipse(img, (cx, cy), (head_w, head_h), 0, 0, 360, skin_color, -1)

    # Forehead hair fringe
    cv2.ellipse(img, (cx, cy - head_h + 30), (head_w + 5, 45), 0, 0, 180, hair_color, -1)

    # Eyes
    eye_dx = 42
    eye_y = cy - 25

    # Eye Sockets (darker for Haar cascade shadow detection)
    cv2.ellipse(img, (cx - eye_dx, eye_y), (22, 14), 0, 0, 360, (skin_color[0]*0.8, skin_color[1]*0.8, skin_color[2]*0.8), -1)
    cv2.ellipse(img, (cx + eye_dx, eye_y), (22, 14), 0, 0, 360, (skin_color[0]*0.8, skin_color[1]*0.8, skin_color[2]*0.8), -1)

    # Eye whites
    cv2.ellipse(img, (cx - eye_dx, eye_y), (16, 9), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (cx + eye_dx, eye_y), (16, 9), 0, 0, 360, (255, 255, 255), -1)

    # Irises & Pupils
    iris_colors = [(120, 60, 20), (30, 80, 30), (40, 40, 40), (140, 90, 40)]
    iris_color = iris_colors[seed_val % len(iris_colors)]
    cv2.circle(img, (cx - eye_dx, eye_y), 7, iris_color, -1)
    cv2.circle(img, (cx + eye_dx, eye_y), 7, iris_color, -1)
    cv2.circle(img, (cx - eye_dx, eye_y), 3, (0, 0, 0), -1)
    cv2.circle(img, (cx + eye_dx, eye_y), 3, (0, 0, 0), -1)

    # Eyebrows (Strong contrast)
    cv2.ellipse(img, (cx - eye_dx, eye_y - 20), (24, 7), -5, 180, 360, hair_color, -1)
    cv2.ellipse(img, (cx + eye_dx, eye_y - 20), (24, 7), 5, 180, 360, hair_color, -1)

    # Nose Bridge & Nostrils
    nose_y = cy + 15
    cv2.line(img, (cx - 2, eye_y), (cx - 8, nose_y), (skin_color[0]*0.7, skin_color[1]*0.7, skin_color[2]*0.7), 2)
    cv2.ellipse(img, (cx, nose_y), (14, 8), 0, 0, 180, (skin_color[0]*0.65, skin_color[1]*0.65, skin_color[2]*0.65), 2)

    # Mouth
    mouth_y = cy + 65
    cv2.ellipse(img, (cx, mouth_y), (30, 12), 0, 0, 180, (60, 60, 180), -1)
    cv2.line(img, (cx - 32, mouth_y), (cx + 32, mouth_y), (40, 40, 120), 2)

    # Variant adjustments (slight lighting/contrast changes)
    if variant > 0:
        factor = 1.0 + (variant * 0.08)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return img


def generate_sample_dataset():
    print("Generating robust sample face image dataset...")

    enrolled_dir = config.ENROLLED_IMAGES_DIR
    test_dir = config.TEST_IMAGES_DIR

    os.makedirs(enrolled_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    identities = ["Alice", "Bob", "Charlie"]
    for name in identities:
        person_dir = os.path.join(enrolled_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        for v in range(3):
            img = draw_realistic_face(name, variant=v)
            img_path = os.path.join(person_dir, f"{name.lower()}_sample_{v+1}.jpg")
            cv2.imwrite(img_path, img)

    # Test query images
    test_cases = [
        ("Alice", 0, "alice_query.jpg"),
        ("Bob", 1, "bob_query.jpg"),
        ("Charlie", 2, "charlie_query.jpg"),
        ("David", 0, "unknown_david_query.jpg"),
    ]
    for name, v, filename in test_cases:
        img = draw_realistic_face(name, variant=v)
        cv2.imwrite(os.path.join(test_dir, filename), img)

    print(f"[OK] Created enrolled sample images in: {enrolled_dir}")
    print(f"[OK] Created query test images in: {test_dir}")


if __name__ == "__main__":
    generate_sample_dataset()
