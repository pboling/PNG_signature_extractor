# PNG Signature Extractor

PNG Signature Extractor removes light backgrounds from images containing dark
signatures and saves the result as a PNG with a transparent background. The
original project described an AI-based approach using the `rembg` library and
its pre-trained U2-Net background-removal model; the current main script uses a
lighter OpenCV pipeline to estimate uneven paper lighting, isolate dark ink
strokes, remove small paper texture artifacts, and render the signature as black
ink on a soft alpha channel.

It is designed for blue or black ink signatures on light paper, including
high-quality screenshots and photos with mild gray shading or uneven lighting.

The script processes every supported image in `Input/` and writes extracted PNG
files to `Output/`. Output filenames append `_extracted` to the original base name.

## Features

- Processes `.png`, `.jpg`, and `.jpeg` files from `Input/`.
- Removes light paper backgrounds and mild paper texture.
- Preserves the signature as black ink with a soft alpha channel, so stroke
  edges may be semi-transparent instead of jagged.
- Saves results as RGBA PNG files in `Output/`.

## Project Layout

```text
Input/                  Place source signature images here
Output/                 Extracted transparent PNG files are written here
signature_extractor.py  Main script
requirements.txt        Python dependencies
```

## Requirements

- Python 3.11 or 3.12 is recommended.
- `opencv-python`
- `numpy<2`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If your system Python cannot create virtual environments because `ensurepip` is
missing, you can use `uv`:

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## Usage

1. Put signature images in `Input/`.
2. Run the extractor:

   ```bash
   python signature_extractor.py
   ```

3. Open the generated files in `Output/`.

Example:

```text
Input/my-signature.png
Output/my-signature_extracted.png
```

## Input Tips

For best results:

- Use a high-resolution image.
- Crop close to the signature before processing.
- Use dark ink on a light background.
- Avoid heavy shadows, ruled paper, complex backgrounds, or very faint ink.

The extractor can handle mild gray paper shading, but cleaner source images will
produce cleaner transparent PNGs.

## License

This project is licensed under the MIT License.

## Author

Yannik Trinn
