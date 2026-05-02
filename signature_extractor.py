#!/usr/bin/env python3
"""
Signature Extractor Script

Extracts dark signature strokes from light paper and saves the result as a PNG
with a transparent background.

Usage examples:
  # Batch mode: process every image in Input/ and write to Output/
  python signature_extractor.py

  # Single file
  python signature_extractor.py --input photo.jpg --output sig.png

  # Custom batch folders, keep the original ink color
  python signature_extractor.py --input-dir scans --output-dir extracted --preserve-color

Assumptions:
  - The signature is dark (black/blue) on a light background.
  - The input has no ruled paper or complex background behind the signature.

Inputs may be 1-channel grayscale, 3-channel BGR, or 4-channel BGRA. RGBA
inputs are flattened onto a white background before processing.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_INPUT_FOLDER = "Input"
DEFAULT_OUTPUT_FOLDER = "Output"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Tuning parameters for shaded-paper extraction.
BACKGROUND_BLUR_SIGMA = 23
DARKNESS_OFFSET = 5.0
DARKNESS_GAIN = 7.5
NOISE_ALPHA_THRESHOLD = 18
MIN_COMPONENT_AREA = 100
SOFTEN_KERNEL_SIZE = (3, 3)


def _load_as_bgr(input_path):
    """Load an image and return it as a 3-channel BGR uint8 array, or None."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        # Flatten RGBA onto a white background so already-transparent inputs
        # do not collapse the local-darkness pipeline.
        bgr = image[:, :, :3].astype(np.float32)
        alpha_in = image[:, :, 3:4].astype(np.float32) / 255.0
        return (bgr * alpha_in + 255.0 * (1.0 - alpha_in)).astype(np.uint8)
    return image


def extract_signature(image, preserve_original_color=False):
    """Run the extraction pipeline on a BGR image and return a BGRA result."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    binary_mask = (alpha_mask > NOISE_ALPHA_THRESHOLD).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    areas = stats[:, cv2.CC_STAT_AREA]
    clean_mask = (areas[labels] >= MIN_COMPONENT_AREA).astype(np.uint8)
    clean_mask[labels == 0] = 0
    alpha_mask = (alpha_mask * clean_mask).astype(np.uint8)
    alpha_mask = cv2.GaussianBlur(alpha_mask, SOFTEN_KERNEL_SIZE, 0)

    if preserve_original_color:
        b, g, r = cv2.split(image)
    else:
        b, g, r = cv2.split(np.zeros_like(image))
    return cv2.merge([b, g, r, alpha_mask])


def process_image(input_path, output_path, preserve_original_color=False):
    """Read input_path, extract the signature, and write to output_path."""
    image = _load_as_bgr(input_path)
    if image is None:
        logging.error(f"Failed to read image: {input_path}")
        return False

    result = extract_signature(image, preserve_original_color=preserve_original_color)

    if not cv2.imwrite(output_path, result):
        logging.error(f"Failed to write image: {output_path}")
        return False
    logging.info(f"Saved extracted image to: {output_path}")
    return True


def _output_path_for(input_path, output_dir):
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, base + "_extracted.png")


def run_batch(input_dir, output_dir, preserve_original_color=False):
    if not os.path.isdir(input_dir):
        logging.error(f"Input folder does not exist: {input_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )
    if not files:
        logging.warning(f"No supported images found in {input_dir}.")
        return 0

    logging.info(f"Found {len(files)} image file(s) in {input_dir}.")
    failures = 0
    for filename in files:
        logging.info(f"Processing image: {filename}")
        in_path = os.path.join(input_dir, filename)
        out_path = _output_path_for(in_path, output_dir)
        if not process_image(in_path, out_path, preserve_original_color):
            failures += 1

    if failures:
        logging.warning(f"Completed with {failures} failure(s).")
        return 1
    logging.info("All images have been processed successfully.")
    return 0


def run_single(input_path, output_path, preserve_original_color=False):
    if not os.path.isfile(input_path):
        logging.error(f"Input file does not exist: {input_path}")
        return 1

    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, exist_ok=True)
    return 0 if process_image(input_path, output_path, preserve_original_color) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract dark signature strokes from light paper as a transparent PNG.",
    )
    parser.add_argument("--input", "-i", help="Path to a single input image.")
    parser.add_argument("--output", "-o", help="Path for the single-file output PNG.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_FOLDER,
        help=f"Folder of input images for batch mode (default: {DEFAULT_INPUT_FOLDER}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Folder for batch-mode output PNGs (default: {DEFAULT_OUTPUT_FOLDER}).",
    )
    parser.add_argument(
        "--preserve-color",
        action="store_true",
        help="Keep the original ink color instead of rendering black.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.input or args.output:
        if not (args.input and args.output):
            logging.error("--input and --output must be used together for single-file mode.")
            return 2
        return run_single(args.input, args.output, args.preserve_color)

    return run_batch(args.input_dir, args.output_dir, args.preserve_color)


if __name__ == "__main__":
    sys.exit(main())
