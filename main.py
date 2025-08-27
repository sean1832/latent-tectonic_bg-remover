#!/usr/bin/env python3
"""
Batch background removal using masks (multithreaded, flexible mask modes).

- Input images are read from INPUT_DIR.
- Mask images are read from MASK_DIR.
- If CROP_SQUARE is True, both inputs and masks are center-cropped to squares.
- If DO_RENAME is True, images in INPUT_DIR and MASK_DIR are renamed to aligned
  sequences (001, 002, ...) so their base names match by order.
- Background selection is configurable:
    * black mode: mask pixels ≤ THRESHOLD are removed
    * white mode: mask pixels ≥ (255 - THRESHOLD) are removed
    * color mode: mask pixels within COLOR_TOLERANCE of COLOR (R,G,B) are removed
- Results are saved as PNG with alpha to OUTPUT_DIR.

Run:
    python main.py [--workers N|max]
                   [--mask-mode black|white|color]
                   [--threshold N]
                   [--color R,G,B --color-tolerance N]
                   [--no-crop] [--no-rename]
                   [--input PATH --mask PATH --output PATH]
                   [--sequence-zero-pad N]

Requires:
    Pillow
    Numpy
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ================================
# DEFAULT FOLDERS
# ================================
OUTPUT_DIR = "output"
INPUT_DIR = "input"
MASK_DIR = "mask"

# ================================
# DEFAULT SETTINGS
# ================================
CROP_SQUARE = True  # Crop both input and mask images into centered squares
DO_RENAME = True  # Rename input and mask images so they have the same base name
SEQUENCE_ZERO_PAD = 3  # Digits in renamed sequence: 3 -> 001, 4 -> 0001

MASK_MODE = "black"  # one of: "black", "white", "color"
BLACK_WHITE_THRESHOLD = 10  # For black/white modes
COLOR_RGB = (0, 0, 0)  # For color mode (R,G,B)
COLOR_TOLERANCE = 15  # For color mode, inclusive distance in RGB

# Worker count: set via CLI; default resolves to os.cpu_count()
WORKERS_DEFAULT = "max"

# ================================
# UTILITIES
# ================================
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist or is not a directory: {directory}")
    files = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    files.sort()
    return files


def temp_move(files: List[Path], temp_dir: Path) -> List[Path]:
    """Move files into a temp directory to avoid rename collisions, return new paths."""
    ensure_directory(temp_dir)
    moved = []
    for file_path in files:
        new_path = temp_dir / file_path.name
        file_path.rename(new_path)
        moved.append(new_path)
    moved.sort()
    return moved


def rename_to_sequence(
    target_dir: Path, files: List[Path], start_number: int = 1, zero_pad: int = 3
) -> None:
    """
    Rename files in-place to a simple increasing sequence with preserved extension.
    Example: 001.png, 002.jpg, ...
    """
    if not files:
        return

    temp_dir = target_dir / "_tmp_renaming"
    moved = temp_move(files, temp_dir)

    for index, file_path in enumerate(moved, start=start_number):
        new_name = f"{str(index).zfill(zero_pad)}{file_path.suffix.lower()}"
        file_path.rename(target_dir / new_name)

    temp_dir.rmdir()


def rename_both_directories(input_dir: Path, mask_dir: Path, zero_pad: int) -> None:
    """
    Rename images in both directories to aligned sequences so base names match by order.
    If counts differ, only the first N pairs will be aligned and the remainder will be left as-is.
    """
    input_files = list_images(input_dir)
    mask_files = list_images(mask_dir)

    if not input_files:
        print("[FATAL] No images found in input directory.", file=sys.stderr)
        sys.exit(2)
    if not mask_files:
        print("[FATAL] No images found in mask directory.", file=sys.stderr)
        sys.exit(2)

    pair_count = min(len(input_files), len(mask_files))
    if pair_count == 0:
        print("[FATAL] No overlapping pairs to rename.", file=sys.stderr)
        sys.exit(2)

    rename_to_sequence(input_dir, input_files[:pair_count], start_number=1, zero_pad=zero_pad)
    rename_to_sequence(mask_dir, mask_files[:pair_count], start_number=1, zero_pad=zero_pad)

    if len(input_files) != len(mask_files):
        print(
            f"[WARN] Different counts — aligned first {pair_count} pairs. "
            f"Unpaired images remain unchanged "
            f"(input: {len(input_files)}, mask: {len(mask_files)})."
        )


def find_images_by_stem(directory: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in list_images(directory):
        mapping[path.stem.lower()] = path
    return mapping


def crop_center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def load_as_rgba(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    return image


def load_mask_rgb_and_l(image_path: Path) -> Tuple[Image.Image, Image.Image]:
    """
    Load mask both as RGB and single channel L.
    If input has alpha, flatten first to avoid surprises.
    """
    mask = Image.open(image_path)
    if mask.mode in ("RGBA", "LA"):
        mask = Image.alpha_composite(
            Image.new("RGBA", mask.size, (0, 0, 0, 255)), mask.convert("RGBA")
        ).convert("RGB")
    elif mask.mode != "RGB":
        mask = mask.convert("RGB")
    mask_l = ImageOps.grayscale(mask)
    return mask, mask_l


def resize_if_needed(
    img: Image.Image, size: Tuple[int, int], hard_edges: bool = True
) -> Image.Image:
    if img.size == size:
        return img
    resample = Image.Resampling.NEAREST if hard_edges else Image.Resampling.BILINEAR
    return img.resize(size, resample=resample)


def combine_alpha(base_rgba: Image.Image, new_alpha: Image.Image) -> Image.Image:
    r, g, b, a = base_rgba.split()
    a_np = np.array(a, dtype=np.uint16)
    new_np = np.array(new_alpha, dtype=np.uint16)
    combined = ((a_np * new_np) // 255).astype(np.uint8)
    return Image.merge("RGBA", (r, g, b, Image.fromarray(combined.astype(np.uint8))))


def alpha_from_mask_black(mask_l: Image.Image, threshold: int) -> Image.Image:
    mask_array = np.array(mask_l, dtype=np.uint8)
    alpha_array = np.where(mask_array <= threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha_array.astype(np.uint8))


def alpha_from_mask_white(mask_l: Image.Image, threshold: int) -> Image.Image:
    mask_array = np.array(mask_l, dtype=np.uint8)
    alpha_array = np.where(mask_array >= 255 - threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha_array.astype(np.uint8))


def alpha_from_mask_color(
    mask_rgb: Image.Image, color: Tuple[int, int, int], tolerance: int
) -> Image.Image:
    """
    Remove pixels whose RGB distance to target color is within tolerance (inclusive).
    Distance is Euclidean in RGB space.
    """
    rgb = np.array(mask_rgb, dtype=np.int16)
    target = np.array(color, dtype=np.int16).reshape(1, 1, 3)
    diff = rgb - target
    dist2 = (diff * diff).sum(axis=2)  # squared distance
    tol2 = tolerance * tolerance
    alpha_array = np.where(dist2 <= tol2, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha_array, mode="L")


def save_png_with_alpha(image_rgba: Image.Image, output_path: Path) -> None:
    image_rgba.save(output_path, format="PNG", optimize=True)


def process_one_pair(
    input_image_path: Path,
    mask_image_path: Path,
    output_dir: Path,
    crop_square: bool,
    mask_mode: str,
    bw_threshold: int,
    color_rgb: Tuple[int, int, int],
    color_tolerance: int,
) -> Optional[Path]:
    try:
        # Load
        input_rgba = load_as_rgba(input_image_path)
        mask_rgb, mask_l = load_mask_rgb_and_l(mask_image_path)

        # Optional cropping (apply the same crop shape to both)
        if crop_square:
            input_rgba = crop_center_square(input_rgba)
            mask_rgb = crop_center_square(mask_rgb)
            mask_l = crop_center_square(mask_l)

        # Align sizes (nearest keeps hard mask edges)
        mask_rgb = resize_if_needed(mask_rgb, input_rgba.size, hard_edges=True)
        mask_l = resize_if_needed(mask_l, input_rgba.size, hard_edges=True)

        # Build alpha based on selected mode
        if mask_mode == "black":
            new_alpha = alpha_from_mask_black(mask_l, bw_threshold)
        elif mask_mode == "white":
            new_alpha = alpha_from_mask_white(mask_l, bw_threshold)
        elif mask_mode == "color":
            new_alpha = alpha_from_mask_color(mask_rgb, color_rgb, color_tolerance)
        else:
            raise ValueError(f"Unsupported mask mode: {mask_mode}")

        # Combine and write
        ensure_directory(output_dir)
        output_name = f"{input_image_path.stem}.png"
        output_path = output_dir / output_name
        out_rgba = combine_alpha(input_rgba, new_alpha)
        save_png_with_alpha(out_rgba, output_path)
        return output_path
    except Exception as exc:
        print(f"[ERROR] Failed on {input_image_path.name}: {exc}", file=sys.stderr)
        return None


def parse_color(text: str) -> Tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be R,G,B")
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Color components must be integers")
    for v in (r, g, b):
        if not (0 <= v <= 255):
            raise argparse.ArgumentTypeError("Color components must be in 0..255")
    return r, g, b


def resolve_workers(value: str) -> int:
    if value.lower() == "max":
        count = os.cpu_count() or 1
        return max(1, count)
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("workers must be an integer or 'max'")
    if n < 1:
        raise argparse.ArgumentTypeError("workers must be ≥ 1")
    return n


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch background removal using masks (multithreaded).")
    p.add_argument("--input", default=INPUT_DIR, help="Input images directory")
    p.add_argument("--mask", default=MASK_DIR, help="Mask images directory")
    p.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--no-crop", action="store_true", help="Disable center square crop")
    p.add_argument("--no-rename", action="store_true", help="Disable renaming to aligned sequences")
    p.add_argument(
        "--sequence-zero-pad", type=int, default=SEQUENCE_ZERO_PAD, help="Digits for sequence names"
    )
    p.add_argument(
        "--workers", default=WORKERS_DEFAULT, help="Number of threads or 'max' (default: max)"
    )
    p.add_argument(
        "--mask-mode",
        choices=["black", "white", "color"],
        default=MASK_MODE,
        help="How mask selects background to remove",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=BLACK_WHITE_THRESHOLD,
        help="Threshold for black/white modes",
    )
    p.add_argument(
        "--color", type=parse_color, default=COLOR_RGB, help="Target color for color mode as R,G,B"
    )
    p.add_argument(
        "--color-tolerance", type=int, default=COLOR_TOLERANCE, help="Tolerance for color mode"
    )
    return p


# ================================
# MAIN
# ================================
def main() -> None:
    args = build_arg_parser().parse_args()

    input_dir = Path(args.input)
    mask_dir = Path(args.mask)
    output_dir = Path(args.output)

    crop_square = not args.no_crop
    do_rename = not args.no_rename
    zero_pad = args.sequence_zero_pad

    mask_mode = args.mask_mode
    bw_threshold = args.threshold
    color_rgb = args.color
    color_tolerance = args.color_tolerance
    workers = resolve_workers(str(args.workers))

    # Optional renaming to align stems
    if do_rename:
        rename_both_directories(input_dir, mask_dir, zero_pad=zero_pad)

    # Build stem maps
    input_map = find_images_by_stem(input_dir)
    mask_map = find_images_by_stem(mask_dir)

    if not input_map:
        print("[FATAL] No supported images found in input directory.", file=sys.stderr)
        sys.exit(2)
    if not mask_map:
        print("[FATAL] No supported images found in mask directory.", file=sys.stderr)
        sys.exit(2)

    # Build work list
    tasks: List[Tuple[Path, Path]] = []
    missing_masks: List[str] = []
    for stem, input_path in input_map.items():
        mask_path = mask_map.get(stem)
        if mask_path is None:
            missing_masks.append(input_path.name)
            continue
        tasks.append((input_path, mask_path))

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        more = " ..." if len(missing_masks) > 10 else ""
        print(f"[WARN] No matching mask for {len(missing_masks)} file(s): {preview}{more}")

    if not tasks:
        print("[FATAL] No pairs to process.", file=sys.stderr)
        sys.exit(2)

    print(
        f"[INFO] Starting with {len(tasks)} pair(s), workers={workers}, "
        f"mode={mask_mode}, threshold={bw_threshold}, color={color_rgb}, tol={color_tolerance}, "
        f"crop_square={crop_square}, rename={do_rename}"
    )

    # Multithreaded processing with progress bar
    success = 0
    ensure_directory(output_dir)

    def _job(pair: Tuple[Path, Path]) -> Optional[Path]:
        in_path, mk_path = pair
        return process_one_pair(
            input_image_path=in_path,
            mask_image_path=mk_path,
            output_dir=output_dir,
            crop_square=crop_square,
            mask_mode=mask_mode,
            bw_threshold=bw_threshold,
            color_rgb=color_rgb,
            color_tolerance=color_tolerance,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, pair) for pair in tasks]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing", unit="img"
        ):
            out_path = future.result()
            if out_path is not None:
                success += 1

    print(f"[DONE] Processed {success}/{len(tasks)} pairs into: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
