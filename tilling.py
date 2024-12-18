import os
import cv2
import numpy as np
from pathlib import Path

# Define paths
input_image_dir = "/Users/nour/Documents/M2/imagerie_biomedical/FCRN/custom/images/train"  # Directory containing original images
input_label_dir = "/Users/nour/Documents/M2/imagerie_biomedical/FCRN/output/heatmaps"  # Directory containing label density images
output_image_dir = "/Users/nour/Documents/M2/imagerie_biomedical/FCRN/augmented/images"  # Directory to save tiled images
output_label_dir = "/Users/nour/Documents/M2/imagerie_biomedical/FCRN/augmented/labels"  # Directory to save tiled labels

# Create output directories if not exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Parameters for tiling
tile_size = 100
stride = 100  # Change to <100 for overlapping tiles

def tile_image(image, tile_size, stride):
    """
    Tiles an image into patches of `tile_size x tile_size` with the given `stride`.
    """
    patches = []
    h, w = image.shape[:2]
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patch = image[y:y + tile_size, x:x + tile_size]
            patches.append((patch, x, y))  # Include coordinates for potential tracking
    return patches

# Process all images and labels
image_paths = sorted(Path(input_image_dir).glob("*.jpg"))  # Adjust extension if needed
label_paths = sorted(Path(input_label_dir).glob("*.jpg"))

if not image_paths or not label_paths:
    print("No images or labels found. Check your input directories or file extensions.")
    exit()

for img_path, lbl_path in zip(image_paths, label_paths):
    print(f"Processing: {img_path.name} and {lbl_path.name}")

    # Load the image and label
    image = cv2.imread(str(img_path))  # RGB image
    label = cv2.imread(str(lbl_path))  

    if image is None or label is None:
        print(f"Failed to load {img_path.name} or {lbl_path.name}. Skipping.")
        continue

    print(f"Image shape: {image.shape}, Label shape: {label.shape}")

    # Ensure both image and label are of the same size
    assert image.shape[:2] == label.shape[:2], f"Size mismatch: {img_path} and {lbl_path}"

    # Tile the image and label
    image_patches = tile_image(image, tile_size, stride)
    label_patches = tile_image(label, tile_size, stride)

    # Sanity check
    assert len(image_patches) == len(label_patches), "Mismatch in patch counts"

    print(f"Generated {len(image_patches)} patches.")

    # Save patches
    base_name = img_path.stem  # File name without extension
    for i, ((img_patch, x, y), (lbl_patch, _, _)) in enumerate(zip(image_patches, label_patches)):
        # Save image patch
        img_patch_path = os.path.join(output_image_dir, f"{base_name}_patch_{i:03d}.png")
        cv2.imwrite(img_patch_path, img_patch)

        # Save corresponding label patch
        lbl_patch_path = os.path.join(output_label_dir, f"{base_name}_patch_{i:03d}.png")
        cv2.imwrite(lbl_patch_path, lbl_patch)

        print(f"Saved patch {i}: Image to {img_patch_path}, Label to {lbl_patch_path}")

print("Tiling complete!")
