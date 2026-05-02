#!/usr/bin/env python3
"""
Signature Extractor Script
This script processes all image files in the "Input" directory, extracts dark
signature strokes from light paper, and saves the result as a PNG with a
transparent background in the "Output" directory.

Assumptions:
  - The signature is dark (black/blue) on a light background.
  - The input has no ruled paper or complex background behind the signature.

Inputs may be 1-channel grayscale, 3-channel BGR, or 4-channel BGRA. RGBA
inputs are flattened onto a white background before processing.
"""

import cv2
import numpy as np
import os
import logging

# Set up logging with a standard format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the input and output directories
INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"

# Tuning parameters for shaded-paper extraction.
BACKGROUND_BLUR_SIGMA = 23
DARKNESS_OFFSET = 5.0
DARKNESS_GAIN = 7.5
NOISE_ALPHA_THRESHOLD = 18
MIN_COMPONENT_AREA = 100
SOFTEN_KERNEL_SIZE = (3, 3)

# When True, keep the source pixel colors so blue ink stays blue.
# When False (default), render the signature as neutral black ink.
PRESERVE_ORIGINAL_COLOR = False


def process_image(filename):
    """
    Processes a single image:
      - Reads the image.
      - Converts it to grayscale.
      - Estimates uneven background lighting.
      - Builds a soft alpha mask from local ink darkness.
      - Removes small connected components caused by paper texture.
      - Saves the final image with transparency.
    """
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + "_extracted.png")

    # Read the image, preserving alpha if present. RGBA inputs (e.g. signatures
    # already on transparent backgrounds) would otherwise read as all-black BGR
    # and collapse the local-darkness pipeline.
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.error(f"Failed to read image: {input_path}")
        return

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha_in = image[:, :, 3:4].astype(np.float32) / 255.0
        image = (bgr * alpha_in + 255.0 * (1.0 - alpha_in)).astype(np.uint8)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate uneven paper/background lighting and isolate strokes by local darkness.
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=BACKGROUND_BLUR_SIGMA,
        sigmaY=BACKGROUND_BLUR_SIGMA,
    )
    darkness = cv2.subtract(background, gray)
    alpha_mask = (
        (darkness.astype(np.float32) - DARKNESS_OFFSET) * DARKNESS_GAIN
    ).clip(0, 255).astype(np.uint8)
    alpha_mask = cv2.GaussianBlur(alpha_mask, SOFTEN_KERNEL_SIZE, 0)

    # Remove small paper-grain artifacts while preserving detached signature marks.
    binary_mask = (alpha_mask > NOISE_ALPHA_THRESHOLD).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    areas = stats[:, cv2.CC_STAT_AREA]
    clean_mask = (areas[labels] >= MIN_COMPONENT_AREA).astype(np.uint8)
    clean_mask[labels == 0] = 0
    alpha_mask = (alpha_mask * clean_mask).astype(np.uint8)
    alpha_mask = cv2.GaussianBlur(alpha_mask, SOFTEN_KERNEL_SIZE, 0)

    # Render the extracted signature. Default is neutral black ink on
    # transparency; PRESERVE_ORIGINAL_COLOR keeps the source pixel colors so
    # blue ink stays blue.
    if PRESERVE_ORIGINAL_COLOR:
        b, g, r = cv2.split(image)
    else:
        b, g, r = cv2.split(np.zeros_like(image))
    # Use soft alpha values so stroke edges stay antialiased instead of jagged.
    alpha = alpha_mask
    # Merge the color channels with the alpha channel to create a BGRA image
    result = cv2.merge([b, g, r, alpha])

    # Save the resulting image as a PNG file (supports transparency)
    cv2.imwrite(output_path, result)
    logging.info(f"Saved extracted image to: {output_path}")


def main():
    """
    Main function:
      - Ensures the output directory exists.
      - Lists all image files in the input directory.
      - Processes each image sequentially.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        logging.info(f"Created output folder: {OUTPUT_FOLDER}")

    files = os.listdir(INPUT_FOLDER)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    logging.info(f"Found {len(image_files)} image file(s) in the input folder.")

    for file in image_files:
        logging.info(f"Processing image: {file}")
        process_image(file)

    logging.info("All images have been processed successfully.")


if __name__ == "__main__":
    main()
