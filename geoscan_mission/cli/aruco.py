#!/usr/bin/env python3
"""Small CLI wrapper around aruco_detector.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from geoscan_mission.vision.aruco import ArucoDetector, DEFAULT_DICTIONARY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect mission ArUco markers in one image.")
    parser.add_argument("--image", default="frame.jpg", help="Input image path.")
    parser.add_argument("--dict", default=DEFAULT_DICTIONARY, help="OpenCV ArUco dictionary name.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--draw", help="Optional output image with detected markers drawn.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        print("cannot read image: {}".format(args.image), file=sys.stderr)
        return 1

    detector = ArucoDetector(dictionary_name=args.dict)
    markers = detector.process_frame(frame)
    result = {
        "image": str(Path(args.image)),
        "dictionary": args.dict,
        "seen_ids": [marker.marker_id for marker in markers],
        "new_ids": [marker.marker_id for marker in markers if marker.first_seen],
        "allowed_ids": detector.found_ids("allowed"),
        "forbidden_ids": detector.found_ids("forbidden"),
        "word": detector.get_word(),
        "markers": [marker.as_dict() for marker in markers],
    }

    if args.draw:
        corners, ids, _ = detector.detect_markers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        ok = cv2.imwrite(args.draw, frame)
        if not ok:
            print("cannot write image: {}".format(args.draw), file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Seen IDs: {}".format(result["seen_ids"]))
        print("Allowed IDs: {}".format(result["allowed_ids"]))
        print("Forbidden IDs: {}".format(result["forbidden_ids"]))
        print("Word: {}".format(result["word"] or "-"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
