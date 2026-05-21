#!/usr/bin/env python3
"""
Localize a downward-looking Mini 2 camera frame on a known 3x3 m map.

Pipeline:
  reference map image -> ORB/SIFT/AKAZE descriptors once
  current frame -> descriptors every frame
  descriptor matching -> cv2.findHomography(frame -> reference)
  frame center on reference -> x/y coordinates in map meters

Examples:
  python3 keypoint_map_localizer.py --reference map_reference.png --image frame.jpg
  python3 keypoint_map_localizer.py --reference map_reference.png --video flight.mp4 --output-dir debug
  python3 keypoint_map_localizer.py --reference map_reference.png --mini2-camera
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_NO_FLY_POLYGONS_M = [
    [(1.05, 0.70), (1.35, 0.70), (1.35, 2.30), (1.05, 2.30)],
    [(1.65, 0.70), (1.95, 0.70), (1.95, 2.30), (1.65, 2.30)],
    [(1.05, 1.35), (1.95, 1.35), (1.95, 1.65), (1.05, 1.65)],
]


@dataclass(frozen=True)
class LocalizationResult:
    ok: bool
    message: str
    x_m: float | None = None
    y_m: float | None = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    center_ref_px: tuple[float, float] | None = None
    homography_frame_to_ref: list[list[float]] | None = None


class KeypointMapLocalizer:
    def __init__(
        self,
        reference_path: str,
        map_width_m: float,
        map_height_m: float,
        method: str,
        nfeatures: int,
        ratio: float,
        min_matches: int,
        min_inliers: int,
        ransac_threshold: float,
    ) -> None:
        self.reference_path = reference_path
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold

        self.ref_gray = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if self.ref_gray is None:
            raise FileNotFoundError(f"cannot read reference map: {reference_path}")

        self.ref_h, self.ref_w = self.ref_gray.shape[:2]
        self.detector, self.norm_type = self._create_detector(method, nfeatures)
        self.matcher = cv2.BFMatcher(self.norm_type)

        self.kp_ref, self.des_ref = self.detector.detectAndCompute(self.ref_gray, None)
        if self.des_ref is None or len(self.kp_ref) < self.min_matches:
            raise RuntimeError(
                f"reference map has too few keypoints: {len(self.kp_ref) if self.kp_ref else 0}"
            )

    @staticmethod
    def _create_detector(method: str, nfeatures: int):
        method = method.lower()

        if method == "orb":
            return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING

        if method == "akaze":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING

        if method == "sift":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("this OpenCV build does not provide SIFT_create()")
            return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2

        raise ValueError("method must be orb, akaze, or sift")

    def ref_pixel_to_map_m(self, px: float, py: float) -> tuple[float, float]:
        x_m = px / self.ref_w * self.map_width_m
        y_m = py / self.ref_h * self.map_height_m
        return float(x_m), float(y_m)

    def map_m_to_ref_pixel(self, x_m: float, y_m: float) -> tuple[int, int]:
        px = int(round(x_m / self.map_width_m * self.ref_w))
        py = int(round(y_m / self.map_height_m * self.ref_h))
        return px, py

    def estimate(self, frame_bgr: np.ndarray) -> LocalizationResult:
        if frame_bgr is None:
            return LocalizationResult(False, "empty frame")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) < self.min_matches:
            return LocalizationResult(
                False,
                "too few frame keypoints",
                good_matches=0,
                inliers=0,
            )

        knn = self.matcher.knnMatch(des_frame, self.des_ref, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            return LocalizationResult(
                False,
                "too few good matches",
                good_matches=len(good),
                inliers=0,
            )

        pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([self.kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            pts_frame,
            pts_ref,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if homography is None or mask is None:
            return LocalizationResult(
                False,
                "homography failed",
                good_matches=len(good),
                inliers=0,
            )

        inliers = int(mask.ravel().sum())
        if inliers < self.min_inliers:
            return LocalizationResult(
                False,
                "too few RANSAC inliers",
                good_matches=len(good),
                inliers=inliers,
                inlier_ratio=inliers / max(len(good), 1),
            )

        h, w = gray.shape[:2]
        center_frame = np.float32([[[w / 2.0, h / 2.0]]])
        center_ref = cv2.perspectiveTransform(center_frame, homography)[0, 0]
        ref_x, ref_y = float(center_ref[0]), float(center_ref[1])

        if not (-0.1 * self.ref_w <= ref_x <= 1.1 * self.ref_w):
            return LocalizationResult(
                False,
                "estimated x is outside reference map",
                good_matches=len(good),
                inliers=inliers,
                inlier_ratio=inliers / max(len(good), 1),
                center_ref_px=(ref_x, ref_y),
            )

        if not (-0.1 * self.ref_h <= ref_y <= 1.1 * self.ref_h):
            return LocalizationResult(
                False,
                "estimated y is outside reference map",
                good_matches=len(good),
                inliers=inliers,
                inlier_ratio=inliers / max(len(good), 1),
                center_ref_px=(ref_x, ref_y),
            )

        x_m, y_m = self.ref_pixel_to_map_m(ref_x, ref_y)
        return LocalizationResult(
            True,
            "ok",
            x_m=x_m,
            y_m=y_m,
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inliers / max(len(good), 1),
            center_ref_px=(ref_x, ref_y),
            homography_frame_to_ref=homography.tolist(),
        )

    def draw_reference_debug(
        self,
        frame_bgr: np.ndarray,
        result: LocalizationResult,
        draw_no_fly: bool,
    ) -> np.ndarray:
        ref_debug = cv2.cvtColor(self.ref_gray, cv2.COLOR_GRAY2BGR)

        if draw_no_fly:
            for polygon_m in DEFAULT_NO_FLY_POLYGONS_M:
                pts = np.array(
                    [self.map_m_to_ref_pixel(x_m, y_m) for x_m, y_m in polygon_m],
                    dtype=np.int32,
                )
                cv2.polylines(ref_debug, [pts], True, (0, 0, 255), 2)

        if result.center_ref_px is not None:
            cx, cy = result.center_ref_px
            cv2.circle(ref_debug, (int(round(cx)), int(round(cy))), 8, (0, 255, 0), -1)

        if result.homography_frame_to_ref is not None:
            H_frame_to_ref = np.array(result.homography_frame_to_ref, dtype=np.float64)
            h, w = frame_bgr.shape[:2]
            frame_corners = np.float32(
                [[[0, 0], [w, 0], [w, h], [0, h]]]
            )
            footprint = cv2.perspectiveTransform(frame_corners, H_frame_to_ref)
            cv2.polylines(ref_debug, [np.int32(footprint[0])], True, (255, 0, 0), 2)

        text = result.message
        if result.ok:
            text = f"x={result.x_m:.2f}m y={result.y_m:.2f}m inliers={result.inliers}"
        cv2.putText(ref_debug, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return ref_debug


def import_pioneer_sdk2():
    try:
        return importlib.import_module("pioneer_sdk2")
    except ModuleNotFoundError as exc:
        if exc.name != "pioneer_sdk2":
            raise
        raise RuntimeError(
            "pioneer_sdk2 is not installed here. Run --mini2-camera on Pioneer Mini 2 / Pioneer OS, "
            "or use --image/--video locally."
        ) from exc


def resolve_sdk2_camera_type(sdk2, camera_type_name: str):
    camera_type_name = camera_type_name.upper()
    if hasattr(sdk2.CameraType, camera_type_name):
        return getattr(sdk2.CameraType, camera_type_name)

    available = sorted(name for name in dir(sdk2.CameraType) if name.isupper())
    raise RuntimeError(
        "SDK2 CameraType.{} is unavailable. Available camera types: {}".format(
            camera_type_name,
            ", ".join(available) if available else "unknown",
        )
    )


def warn_show_disabled(show: bool) -> None:
    if show:
        print(
            "WARNING: --show is ignored because this may run on OpenCV without HighGUI. "
            "Use --output-dir for debug images instead.",
            file=sys.stderr,
        )


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def result_to_row(frame_id: int, result: LocalizationResult) -> dict[str, object]:
    row = asdict(result)
    row.pop("homography_frame_to_ref", None)
    row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
    row["frame_id"] = frame_id
    return row


def print_result(frame_id: int, result: LocalizationResult) -> None:
    data = result_to_row(frame_id, result)
    print(json.dumps(data, ensure_ascii=False))


def handle_frame(
    localizer: KeypointMapLocalizer,
    frame_bgr: np.ndarray,
    frame_id: int,
    args: argparse.Namespace,
) -> LocalizationResult:
    result = localizer.estimate(frame_bgr)
    print_result(frame_id, result)

    if args.csv:
        append_csv(Path(args.csv), result_to_row(frame_id, result))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ref_debug = localizer.draw_reference_debug(frame_bgr, result, args.draw_no_fly)
        cv2.imwrite(str(out_dir / f"ref_debug_{frame_id:06d}.jpg"), ref_debug)

    return result


def iter_video_frames(path: str, frame_step: int) -> Iterable[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")

    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id % frame_step == 0:
                yield frame_id, frame
            frame_id += 1
    finally:
        capture.release()


def run_image(localizer: KeypointMapLocalizer, args: argparse.Namespace) -> int:
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {args.image}")
    handle_frame(localizer, frame, 0, args)
    return 0


def run_video(localizer: KeypointMapLocalizer, args: argparse.Namespace) -> int:
    for frame_id, frame in iter_video_frames(args.video, args.frame_step):
        handle_frame(localizer, frame, frame_id, args)
    return 0


def run_mini2_camera(localizer: KeypointMapLocalizer, args: argparse.Namespace) -> int:
    sdk2 = import_pioneer_sdk2()
    camera_type = resolve_sdk2_camera_type(sdk2, args.sdk2_camera_type)
    try:
        camera = sdk2.Camera(camera_type=camera_type)
    except TypeError:
        camera = sdk2.Camera(camera_type)

    try:
        frame_id = 0
        while args.max_frames <= 0 or frame_id < args.max_frames:
            frame = camera.get_cv_frame(timeout=2.0)
            if frame is None:
                print("empty camera frame", file=sys.stderr)
                continue

            handle_frame(localizer, frame, frame_id, args)
            frame_id += 1

            time.sleep(args.frame_interval)
    finally:
        camera.stop()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ORB/SIFT/AKAZE map localization through homography.",
    )
    parser.add_argument("--reference", required=True, help="Reference image of the full 3x3 m map.")
    parser.add_argument("--map-width-m", type=float, default=3.0)
    parser.add_argument("--map-height-m", type=float, default=3.0)
    parser.add_argument("--method", choices=["orb", "sift", "akaze"], default="orb")
    parser.add_argument("--nfeatures", type=int, default=3000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Single drone frame.")
    source.add_argument("--video", help="Video file.")
    source.add_argument("--mini2-camera", action="store_true", help="Read frames from Mini 2 Camera.")

    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument("--frame-interval", type=float, default=0.05, help="Delay between Mini 2 camera frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Mini 2 camera frame limit. 0 means unlimited.")
    parser.add_argument("--sdk2-camera-type", default="OPT", help="SDK2 CameraType name for --mini2-camera.")
    parser.add_argument("--csv", help="Optional CSV log path.")
    parser.add_argument("--output-dir", help="Optional directory for debug images.")
    parser.add_argument("--show", action="store_true", help="Deprecated: ignored on headless/OpenCV-no-GUI builds.")
    parser.add_argument("--draw-no-fly", action="store_true", help="Draw built-in H-shaped no-fly polygons.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_show_disabled(args.show)

    localizer = KeypointMapLocalizer(
        reference_path=args.reference,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        method=args.method,
        nfeatures=args.nfeatures,
        ratio=args.ratio,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        ransac_threshold=args.ransac_threshold,
    )

    if args.image:
        return run_image(localizer, args)
    if args.video:
        return run_video(localizer, args)
    if args.mini2_camera:
        return run_mini2_camera(localizer, args)

    parser.error("no input source selected")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
