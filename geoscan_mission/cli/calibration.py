#!/usr/bin/env python3
"""CLI for headless Mini 2 camera calibration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from geoscan_mission.flight.camera import OpenCvCamera, Sdk2Camera
from geoscan_mission.flight.control import import_pioneer_sdk2
from geoscan_mission.vision.calibration import (
    DEFAULT_BOARD_SIZE,
    CameraCalibration,
    calibrate_camera,
    get_images_from_folder,
    save_camera_calibration,
)


def create_camera(args: argparse.Namespace):
    if args.camera_source == "opencv":
        return OpenCvCamera(args.camera_index), "opencv:{}".format(args.camera_index)

    sdk2 = import_pioneer_sdk2()
    return Sdk2Camera(sdk2, args.sdk2_camera_type, args.camera_timeout), args.sdk2_camera_type.upper()


def maybe_save_frame(frame: np.ndarray, frames_dir: Path | None, index: int) -> None:
    if frames_dir is None:
        return

    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / "frame_{:04d}.jpg".format(index)
    if not cv2.imwrite(str(path), frame):
        print("WARNING: failed to save {}".format(path), file=sys.stderr)


def capture_frames(args: argparse.Namespace) -> list[np.ndarray]:
    camera, camera_name = create_camera(args)
    frames_dir = None if args.no_save_frames else Path(args.frames_dir)
    frames: list[np.ndarray] = []

    try:
        print("Camera source: {}".format(camera_name))
        if args.warmup_frames > 0:
            print("Skipping {} warmup frames...".format(args.warmup_frames))
        for _ in range(args.warmup_frames):
            camera.read()
            time.sleep(0.05)

        print(
            "Capturing {} frames every {:.2f}s. Move/tilt the checkerboard between frames.".format(
                args.max_frames,
                args.capture_interval,
            )
        )
        while len(frames) < args.max_frames:
            frame = camera.read()
            if frame is None:
                print("WARNING: empty camera frame", file=sys.stderr)
                time.sleep(0.1)
                continue

            frame_index = len(frames) + 1
            frames.append(frame.copy())
            maybe_save_frame(frame, frames_dir, frame_index)
            print("Captured frame {}/{}".format(frame_index, args.max_frames))
            if frame_index < args.max_frames and args.capture_interval > 0:
                time.sleep(args.capture_interval)

    finally:
        camera.close()

    return frames


def print_summary(calibration: CameraCalibration, output_path: str) -> None:
    print("Calibration saved: {}".format(output_path))
    print("Valid chessboard frames: {}/{}".format(calibration.valid_frames, calibration.total_frames))
    if calibration.image_size is not None:
        print("Image size: {}x{}".format(calibration.image_size[0], calibration.image_size[1]))
    if calibration.rms is not None:
        print("RMS: {:.6f}".format(calibration.rms))
    if calibration.reprojection_error is not None:
        print("Mean reprojection error: {:.6f}".format(calibration.reprojection_error))
    print("Camera matrix:")
    print(calibration.camera_matrix)
    print("Distortion coefficients:")
    print(calibration.dist_coeffs)


def run(args: argparse.Namespace) -> int:
    if args.images:
        images = get_images_from_folder(args.images, args.glob)
        print("Loaded {} images from {}".format(len(images), args.images))
    else:
        images = capture_frames(args)

    calibration = calibrate_camera(
        images,
        board_size=(args.board_cols, args.board_rows),
        square_size=args.square_size,
        show=args.show,
        debug_dir=args.debug_dir,
    )
    if calibration.valid_frames is not None and calibration.valid_frames < args.min_valid_frames:
        raise RuntimeError(
            "only {} valid chessboard frames found, need at least {}; "
            "collect more varied board angles or lower --min-valid-frames".format(
                calibration.valid_frames,
                args.min_valid_frames,
            )
        )
    save_camera_calibration(calibration, args.output)
    print_summary(calibration, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from chessboard images or a Mini 2 SDK2 camera.",
    )
    parser.add_argument("--images", help="Folder with calibration images. If omitted, frames are captured from a camera.")
    parser.add_argument("--glob", default="*.jpg", help="Glob pattern inside --images.")
    parser.add_argument("--camera-source", choices=["sdk2", "opencv"], default="sdk2")
    parser.add_argument("--sdk2-camera-type", default="OPT")
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=25)
    parser.add_argument("--capture-interval", type=float, default=1.0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--frames-dir", default="calibration_frames")
    parser.add_argument("--no-save-frames", action="store_true", help="Do not save captured source frames.")
    parser.add_argument("--output", default="data.yml")
    parser.add_argument("--debug-dir", help="Save annotated chessboard detections here.")
    parser.add_argument("--min-valid-frames", type=int, default=8)
    parser.add_argument("--board-cols", type=int, default=DEFAULT_BOARD_SIZE[0], help="Inner chessboard corners by columns.")
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD_SIZE[1], help="Inner chessboard corners by rows.")
    parser.add_argument("--square-size", type=float, default=1.0, help="Physical square size in any consistent unit.")
    parser.add_argument("--show", action="store_true", help="Show frames with detected corners. Local desktop only.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.board_cols < 2 or args.board_rows < 2:
        raise ValueError("--board-cols and --board-rows must be at least 2")
    if args.square_size <= 0:
        raise ValueError("--square-size must be positive")
    if args.camera_timeout <= 0:
        raise ValueError("--camera-timeout must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.capture_interval < 0:
        raise ValueError("--capture-interval must be >= 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.min_valid_frames <= 0:
        raise ValueError("--min-valid-frames must be positive")
    if not args.sdk2_camera_type.strip():
        raise ValueError("--sdk2-camera-type must not be empty")
    if args.images and not Path(args.images).exists():
        raise ValueError("--images folder does not exist: {}".format(args.images))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        return run(args)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
