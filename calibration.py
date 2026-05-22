#!/usr/bin/env python3
"""Compatibility wrapper for the modular camera calibration CLI."""

from geoscan_mission.cli.calibration import build_parser, capture_frames, main, run, validate_args
from geoscan_mission.vision.calibration import (
    DEFAULT_BOARD_SIZE,
    CameraCalibration,
    calibrate,
    calibrate_camera,
    get_images_from_folder,
    load_camera_calibration,
    load_coefficients,
    save_camera_calibration,
    save_coefficients,
)


if __name__ == "__main__":
    raise SystemExit(main())
