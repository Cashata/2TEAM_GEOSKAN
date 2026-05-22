#!/usr/bin/env python3
"""Compatibility wrapper and demo for the packaged ORB grid detector."""

from __future__ import annotations

import argparse

import cv2

from geoscan_mission.vision.orb_grid import MAP_FILE, OrbDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview ORB grid localization on a video/image source.")
    parser.add_argument("--reference", default=MAP_FILE)
    parser.add_argument("--video", default="camera_overlay.avi")
    parser.add_argument("--map-width-m", type=float, default=3.0)
    parser.add_argument("--map-height-m", type=float, default=3.0)
    parser.add_argument("--show-keypoints", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detector = OrbDetector(
        map_file=args.reference,
        map_size_m=(args.map_width_m, args.map_height_m),
        show_keypoints=args.show_keypoints,
    )
    video = cv2.VideoCapture(args.video)

    try:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            cv2.imshow("AVI Frame", frame)
            cv2.imshow("ORB Frame", detector.draw_debug(frame))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
