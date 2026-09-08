"""
Utility to generate synthetic realistic reference photos for healthy and damaged conveyor belt joints.
"""

import os
import cv2
import numpy as np


def generate_fixtures():
    np.random.seed(42)
    target_dir = os.path.join(os.path.dirname(__file__), "test_images")
    os.makedirs(target_dir, exist_ok=True)

    width, height = 640, 480
    bg_color = (25, 25, 25)  # Dark rubber belt

    # Joint patch dimensions
    jx, jy, jw, jh = 180, 140, 280, 200

    # -------------------------------------------------------------------------
    # 1. Healthy Sample: Flat, smooth, metallic gradient, faint horizontal texture
    # -------------------------------------------------------------------------
    healthy_img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Base metallic foil (high brightness, light gray/silver)
    foil_base = np.full((jh, jw, 3), (220, 220, 220), dtype=np.uint8)

    # Smooth specular gradient
    gradient = np.linspace(200, 240, jw).astype(np.uint8)
    for c in range(3):
        foil_base[:, :, c] = gradient[None, :]

    # Fine horizontal brushed metal scratches (subtle)
    noise = np.random.normal(0, 3, (jh, jw)).astype(np.float32)
    for c in range(3):
        foil_base[:, :, c] = np.clip(foil_base[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Apply subtle horizontal blur to simulate brushed metal
    foil_base = cv2.blur(foil_base, (7, 1))

    healthy_img[jy:jy+jh, jx:jx+jw] = foil_base
    healthy_path = os.path.join(target_dir, "healthy_sample.jpg")
    cv2.imwrite(healthy_path, healthy_img)
    print(f"[GENERATED] Healthy sample -> {healthy_path}")

    # -------------------------------------------------------------------------
    # 2. Damaged Sample: Crumpled foil, sharp diagonal fold lines, specular patchiness
    # -------------------------------------------------------------------------
    damage_img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    foil_damage = foil_base.copy()

    # Add strong diagonal crease lines (bright specular edge & dark shadow fold line)
    # Line 1: Strong diagonal fold from (30, 20) to (250, 180)
    cv2.line(foil_damage, (30, 20), (250, 180), (30, 30, 30), 4)       # Dark shadow crease
    cv2.line(foil_damage, (33, 18), (253, 178), (255, 255, 255), 3)   # High specular reflection line

    # Line 2: Secondary diagonal branch crease from (150, 40) to (80, 160)
    cv2.line(foil_damage, (150, 40), (80, 160), (20, 20, 20), 3)
    cv2.line(foil_damage, (152, 38), (82, 158), (250, 250, 250), 2)

    # Add patchy irregular specular highlights & shadow blotches (crumple texture)
    for _ in range(8):
        cx_spot = np.random.randint(40, jw - 40)
        cy_spot = np.random.randint(30, jh - 30)
        axes = (np.random.randint(15, 45), np.random.randint(10, 30))
        angle = np.random.randint(0, 180)
        val = 255 if np.random.rand() > 0.5 else 40
        cv2.ellipse(foil_damage, (cx_spot, cy_spot), axes, angle, 0, 360, (val, val, val), -1)

    # Blur slightly so patchiness has continuous gradients & edges
    foil_damage = cv2.GaussianBlur(foil_damage, (5, 5), 0)

    damage_img[jy:jy+jh, jx:jx+jw] = foil_damage
    damage_path = os.path.join(target_dir, "damage_sample.jpg")
    cv2.imwrite(damage_path, damage_img)
    print(f"[GENERATED] Damage sample -> {damage_path}")


if __name__ == "__main__":
    generate_fixtures()
