#!/usr/bin/env python3
"""
Convert a large TIFF map into a JPEG reference image for ORB/RANSAC.

Default usage from the repository root:
  python tools/convert_map_tif.py

This auto-detects the largest *.tif/*.tiff file in the current directory and
writes map.jpg. Use --max-side to control the output size.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_OUTPUT = Path("map.jpg")
DEFAULT_MAX_SIDE = 6000
DEFAULT_QUALITY = 92


def find_input_file() -> Path:
    candidates = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        candidates.extend(Path(".").glob(pattern))

    unique = sorted({path.resolve(): path for path in candidates}.values(), key=lambda p: p.stat().st_size, reverse=True)
    if not unique:
        raise SystemExit("No TIFF file found. Pass the input path explicitly.")
    return unique[0]


def target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0 or max(width, height) <= max_side:
        return width, height

    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def select_pillow_frame(image, requested_frame: int | None) -> int:
    n_frames = int(getattr(image, "n_frames", 1))
    if requested_frame is not None:
        if requested_frame < 0 or requested_frame >= n_frames:
            raise SystemExit(f"Frame index {requested_frame} is outside TIFF frame range 0..{n_frames - 1}.")
        image.seek(requested_frame)
        return requested_frame

    best_index = 0
    best_area = -1
    for index in range(n_frames):
        image.seek(index)
        width, height = image.size
        area = width * height
        if area > best_area:
            best_area = area
            best_index = index

    image.seek(best_index)
    return best_index


def save_with_pillow(src: Path, dst: Path, max_side: int, quality: int, frame: int | None) -> dict[str, object]:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None

    with Image.open(src) as image:
        selected_frame = select_pillow_frame(image, frame)
        image = ImageOps.exif_transpose(image)
        source_size = image.size
        output_size = target_size(*source_size, max_side)

        if output_size != source_size:
            image.thumbnail(output_size, Image.Resampling.LANCZOS, reducing_gap=3.0)

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        tmp = dst.with_name(f".{dst.name}.tmp")
        image.save(
            tmp,
            "JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=0,
        )
        os.replace(tmp, dst)

        return {
            "backend": "pillow",
            "frame": selected_frame,
            "source_size": source_size,
            "output_size": image.size,
        }


def save_with_opencv(src: Path, dst: Path, max_side: int, quality: int) -> dict[str, object]:
    import cv2
    import numpy as np

    data = np.fromfile(str(src), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {src}")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width = image.shape[:2]
    output_width, output_height = target_size(width, height, max_side)
    if (output_width, output_height) != (width, height):
        image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("OpenCV could not encode JPEG output")
    encoded.tofile(str(dst))

    return {
        "backend": "opencv",
        "frame": 0,
        "source_size": (width, height),
        "output_size": (output_width, output_height),
    }


def convert_map(args: argparse.Namespace) -> dict[str, object]:
    src = args.input or find_input_file()
    dst = args.output

    if not src.exists():
        raise SystemExit(f"Input file does not exist: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if args.backend in ("auto", "pillow"):
        try:
            return save_with_pillow(src, dst, args.max_side, args.quality, args.frame)
        except ImportError:
            if args.backend == "pillow":
                raise
        except Exception:
            if args.backend == "pillow":
                raise
            raise

    return save_with_opencv(src, dst, args.max_side, args.quality)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Input TIFF map. Defaults to the largest TIFF in cwd.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JPEG path.")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="Maximum output width/height in pixels. Use 0 to keep full size.")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality, 1..100.")
    parser.add_argument("--frame", type=int, default=None, help="TIFF frame/page to use. Defaults to the largest frame.")
    parser.add_argument("--backend", choices=("auto", "pillow", "opencv"), default="auto", help="Image backend.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in range 1..100")

    info = convert_map(args)
    print(f"backend: {info['backend']}")
    print(f"frame: {info['frame']}")
    print(f"source size: {info['source_size'][0]} x {info['source_size'][1]}")
    print(f"output size: {info['output_size'][0]} x {info['output_size'][1]}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
