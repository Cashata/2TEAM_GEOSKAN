#!/usr/bin/env python3
"""
Fly local waypoints and localize camera frames on a known 3x3 m map.

This combines two ideas:
  1. Pioneer local waypoint flight.
  2. ORB/AKAZE/SIFT feature matching against map.jpg with homography + RANSAC.

Default behavior:
  - reference map: map.jpg
  - camera: Pioneer-SDK2 Camera.get_cv_frame() from CameraType.OPT
  - waypoints: a small square relative to the takeoff point

Local camera/localization test without drone:
  python fly_orb_ransac.py --no-flight --reference map.jpg --camera-index 0

Real flight:
  python3 fly_orb_ransac.py --reference map.jpg
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_WAYPOINTS = [
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.5),
    (0.0, 0.0, 1.0),
]

LAND_COMMANDS = {
    "land",
    "stop",
    "q",
    "quit",
    "exit",
    "posadka",
    "sest",
    "посадка",
    "сесть",
    "стоп",
}


@dataclass
class LocalizeResult:
    ok: bool
    message: str
    x_m: float | None = None
    y_m: float | None = None
    raw_x_m: float | None = None
    raw_y_m: float | None = None
    smooth_x_m: float | None = None
    smooth_y_m: float | None = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    homography_area: float | None = None
    accepted_by_filter: bool = False
    filter_reason: str = ""


class OrbRansacLocalizer:
    def __init__(
        self,
        reference_path: str,
        map_width_m: float,
        map_height_m: float,
        feature: str,
        nfeatures: int,
        ratio: float,
        min_matches: int,
        min_inliers: int,
        ransac_threshold: float,
        frame_width: int,
        frame_height: int,
        reference_max_size: int,
        clahe_clip: float,
        clahe_tile: int,
        min_inlier_ratio: float,
        min_homography_area_m2: float,
        max_homography_area_m2: float,
        max_position_jump: float,
        ema_alpha: float,
    ) -> None:
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.feature = feature.lower()
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.frame_size = (frame_width, frame_height)
        self.min_inlier_ratio = min_inlier_ratio
        self.min_homography_area_m2 = min_homography_area_m2
        self.max_homography_area_m2 = (
            max_homography_area_m2 if max_homography_area_m2 > 0 else map_width_m * map_height_m * 1.1
        )
        self.max_position_jump = max_position_jump
        self.ema_alpha = ema_alpha
        self.smooth_x_m: float | None = None
        self.smooth_y_m: float | None = None
        self.first_valid_for_waypoint = True
        self.clahe = (
            cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
            if clahe_clip > 0
            else None
        )

        self.reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if self.reference is None:
            raise FileNotFoundError("Cannot read reference map: {}".format(reference_path))

        self.reference = self.resize_reference(self.reference, reference_max_size)
        self.ref_h, self.ref_w = self.reference.shape[:2]
        self.detector, self.norm_type = self.create_detector(self.feature, nfeatures)
        reference_features = self.preprocess_gray(self.reference)
        self.kp_ref, self.des_ref = self.detector.detectAndCompute(reference_features, None)
        if self.des_ref is None or len(self.kp_ref) < min_matches:
            raise RuntimeError("Reference map has too few {} keypoints".format(self.feature.upper()))

        self.matcher = cv2.BFMatcher(self.norm_type)

    @staticmethod
    def create_detector(feature: str, nfeatures: int):
        if feature == "orb":
            return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING
        if feature == "akaze":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        if feature == "sift":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("OpenCV build does not provide SIFT_create(); use --feature orb or akaze")
            return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2
        raise ValueError("--feature must be orb, akaze, or sift")

    @staticmethod
    def resize_reference(reference: np.ndarray, max_size: int) -> np.ndarray:
        if max_size <= 0:
            return reference
        height, width = reference.shape[:2]
        longest = max(width, height)
        if longest <= max_size:
            return reference
        scale = max_size / float(longest)
        return cv2.resize(reference, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)

    def preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        if self.clahe is None:
            return gray
        return self.clahe.apply(gray)

    def prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        width, height = self.frame_size
        if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
            return frame_bgr
        return cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_AREA)

    def start_waypoint(self) -> None:
        self.first_valid_for_waypoint = True

    def ref_pixel_to_map_m(self, px: float, py: float) -> tuple[float, float]:
        x_m = px / self.ref_w * self.map_width_m
        y_m = py / self.ref_h * self.map_height_m
        return float(x_m), float(y_m)

    def homography_area_m2(self, homography: np.ndarray, frame_shape: tuple[int, int]) -> float | None:
        height, width = frame_shape
        corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
        projected = cv2.perspectiveTransform(corners, homography)[0]
        if not np.all(np.isfinite(projected)):
            return None
        area_px = abs(cv2.contourArea(projected.astype(np.float32)))
        pixel_area_m2 = (self.map_width_m / self.ref_w) * (self.map_height_m / self.ref_h)
        return float(area_px * pixel_area_m2)

    def last_smooth(self) -> tuple[float | None, float | None]:
        return self.smooth_x_m, self.smooth_y_m

    def reject_result(
        self,
        message: str,
        good_matches: int = 0,
        inliers: int = 0,
        inlier_ratio: float = 0.0,
        homography_area: float | None = None,
        raw_x_m: float | None = None,
        raw_y_m: float | None = None,
    ) -> LocalizeResult:
        return LocalizeResult(
            False,
            message,
            raw_x_m=raw_x_m,
            raw_y_m=raw_y_m,
            smooth_x_m=self.smooth_x_m,
            smooth_y_m=self.smooth_y_m,
            good_matches=good_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            homography_area=homography_area,
            accepted_by_filter=False,
            filter_reason=message,
        )

    def accept_result(
        self,
        raw_x_m: float,
        raw_y_m: float,
        good_matches: int,
        inliers: int,
        inlier_ratio: float,
        homography_area: float,
    ) -> LocalizeResult:
        if self.smooth_x_m is None or self.smooth_y_m is None:
            smooth_x_m = raw_x_m
            smooth_y_m = raw_y_m
        else:
            smooth_x_m = (1.0 - self.ema_alpha) * self.smooth_x_m + self.ema_alpha * raw_x_m
            smooth_y_m = (1.0 - self.ema_alpha) * self.smooth_y_m + self.ema_alpha * raw_y_m

        self.smooth_x_m = smooth_x_m
        self.smooth_y_m = smooth_y_m
        self.first_valid_for_waypoint = False

        return LocalizeResult(
            True,
            "ok",
            x_m=smooth_x_m,
            y_m=smooth_y_m,
            raw_x_m=raw_x_m,
            raw_y_m=raw_y_m,
            smooth_x_m=smooth_x_m,
            smooth_y_m=smooth_y_m,
            good_matches=good_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            homography_area=homography_area,
            accepted_by_filter=True,
            filter_reason="ok",
        )

    def estimate(self, frame_bgr: np.ndarray) -> tuple[LocalizeResult, np.ndarray | None, np.ndarray]:
        frame_bgr = self.prepare_frame(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.preprocess_gray(gray)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) < self.min_matches:
            return self.reject_result("too few frame keypoints"), None, frame_bgr

        knn = self.matcher.knnMatch(des_frame, self.des_ref, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            return self.reject_result("too few good matches", good_matches=len(good)), None, frame_bgr

        pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([self.kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            pts_frame,
            pts_ref,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if homography is None or mask is None:
            return self.reject_result("RANSAC homography failed", good_matches=len(good)), None, frame_bgr

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / max(len(good), 1)
        homography_area = self.homography_area_m2(homography, gray.shape[:2])

        if inliers < self.min_inliers:
            return (
                self.reject_result(
                    "too few RANSAC inliers",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                ),
                homography,
                frame_bgr,
            )

        if inlier_ratio < self.min_inlier_ratio:
            return (
                self.reject_result(
                    "low RANSAC inlier ratio",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                ),
                homography,
                frame_bgr,
            )

        if homography_area is None or homography_area < self.min_homography_area_m2:
            return (
                self.reject_result(
                    "homography area too small",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                ),
                homography,
                frame_bgr,
            )

        if homography_area > self.max_homography_area_m2:
            return (
                self.reject_result(
                    "homography area too large",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                ),
                homography,
                frame_bgr,
            )

        h, w = gray.shape[:2]
        center_frame = np.float32([[[w / 2.0, h / 2.0]]])
        center_ref = cv2.perspectiveTransform(center_frame, homography)[0, 0]
        ref_x, ref_y = float(center_ref[0]), float(center_ref[1])

        if not (0.0 <= ref_x <= self.ref_w and 0.0 <= ref_y <= self.ref_h):
            return (
                self.reject_result(
                    "estimated center is outside reference map",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                ),
                homography,
                frame_bgr,
            )

        raw_x_m, raw_y_m = self.ref_pixel_to_map_m(ref_x, ref_y)
        if (
            self.max_position_jump > 0
            and not self.first_valid_for_waypoint
            and self.smooth_x_m is not None
            and self.smooth_y_m is not None
            and math.dist((raw_x_m, raw_y_m), (self.smooth_x_m, self.smooth_y_m)) > self.max_position_jump
        ):
            return (
                self.reject_result(
                    "position jump too large",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                    raw_x_m=raw_x_m,
                    raw_y_m=raw_y_m,
                ),
                homography,
                frame_bgr,
            )

        return self.accept_result(raw_x_m, raw_y_m, len(good), inliers, inlier_ratio, homography_area), homography, frame_bgr

    def draw_debug(self, frame_bgr: np.ndarray, result: LocalizeResult, homography: np.ndarray | None) -> np.ndarray:
        ref_debug = cv2.cvtColor(self.reference, cv2.COLOR_GRAY2BGR)

        if homography is not None:
            h, w = frame_bgr.shape[:2]
            corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
            projected = cv2.perspectiveTransform(corners, homography)
            if np.all(np.isfinite(projected)):
                cv2.polylines(ref_debug, [np.int32(projected[0])], True, (255, 0, 0), 2)

        point_x = result.x_m if result.ok else result.raw_x_m
        point_y = result.y_m if result.ok else result.raw_y_m
        if point_x is not None and point_y is not None:
            px = int(round(point_x / self.map_width_m * self.ref_w))
            py = int(round(point_y / self.map_height_m * self.ref_h))
            color = (0, 255, 0) if result.ok else (0, 0, 255)
            cv2.circle(ref_debug, (px, py), 8, color, -1)

        if result.ok and result.x_m is not None and result.y_m is not None:
            text = "x={:.2f}m y={:.2f}m inliers={}".format(result.x_m, result.y_m, result.inliers)
        else:
            text = "{} matches={} inliers={}".format(result.message, result.good_matches, result.inliers)

        cv2.putText(ref_debug, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return ref_debug


class OpenCvCamera:
    def __init__(self, camera_index: int) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open OpenCV camera index {}".format(camera_index))

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class Sdk2Camera:
    def __init__(self, sdk2, camera_type_name: str, timeout: float) -> None:
        camera_type = resolve_sdk2_camera_type(sdk2, camera_type_name)
        try:
            self.camera = sdk2.Camera(camera_type=camera_type)
        except TypeError:
            self.camera = sdk2.Camera(camera_type)
        self.timeout = timeout

    def read(self) -> np.ndarray | None:
        return self.camera.get_cv_frame(timeout=self.timeout)

    def close(self) -> None:
        if hasattr(self.camera, "close"):
            self.camera.close()


def import_pioneer_sdk2():
    try:
        return importlib.import_module("pioneer_sdk2")
    except ModuleNotFoundError as exc:
        if exc.name != "pioneer_sdk2":
            raise
        raise RuntimeError(
            "pioneer_sdk2 is not installed in this Python environment. "
            "Run real flights on Pioneer Mini 2 / Pioneer OS, or use --no-flight locally."
        ) from exc


def create_pioneer(sdk2):
    try:
        return sdk2.Pioneer(wait_callback=True, safety_command=True)
    except TypeError:
        return sdk2.Pioneer()


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
            "WARNING: --show is ignored here because onboard OpenCV may be built without HighGUI. "
            "Use --debug-dir to save debug images instead.",
            file=sys.stderr,
        )


def parse_waypoint(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("waypoint must be x,y,z")
    return parts[0], parts[1], parts[2]


def parse_float_list(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("list must contain at least one number")
    return parts


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def build_lawnmower_points(area_size: float, margin: float, grid_size: int, height: float) -> list[tuple[float, float, float]]:
    lo = -area_size / 2.0 + margin
    hi = area_size / 2.0 - margin
    xs = linspace(lo, hi, grid_size)
    ys = linspace(lo, hi, grid_size)

    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((round(x, 3), round(y, 3), height))
    return points


def build_square_points(area_size: float, margin: float, height: float) -> list[tuple[float, float, float]]:
    lo = -area_size / 2.0 + margin
    hi = area_size / 2.0 - margin
    return [
        (lo, lo, height),
        (hi, lo, height),
        (hi, hi, height),
        (lo, hi, height),
        (lo, lo, height),
    ]


def resolve_waypoints(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    if args.waypoint:
        return args.waypoint

    if args.trajectory == "waypoints":
        return list(DEFAULT_WAYPOINTS)

    if args.trajectory == "square":
        return build_square_points(args.area_size, args.margin, args.height)

    if args.trajectory == "lawnmower":
        return build_lawnmower_points(args.area_size, args.margin, args.grid_size, args.height)

    if args.trajectory == "cube":
        points = []
        layers = args.layers or [args.height, args.high_height]
        for layer_index, height in enumerate(layers):
            layer = build_lawnmower_points(args.area_size, args.margin, args.grid_size, height)
            if layer_index % 2:
                layer = list(reversed(layer))
            points.extend(layer)
        return points

    raise ValueError("unknown trajectory: {}".format(args.trajectory))


def start_command_listener(stop_event: threading.Event) -> threading.Thread:
    def listen() -> None:
        print("Type 'land', 'stop', 'q', 'posadka' or 'sest' + Enter for graceful landing.")
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except OSError:
                return
            if line == "":
                return

            command = line.strip().lower()
            if not command:
                continue
            if command in LAND_COMMANDS:
                print("Graceful landing requested by command: {}".format(command))
                stop_event.set()
                return
            print("Unknown command '{}'. Use: {}".format(command, ", ".join(sorted(LAND_COMMANDS))))

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return thread


def append_csv(path: str | None, row: dict[str, object]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def result_row(
    point_index: int,
    frame_id: int,
    result: LocalizeResult,
    camera_type: str,
    target_point: tuple[float, float, float],
    yaw: float,
) -> dict[str, object]:
    row = asdict(result)
    row["timestamp"] = time.time()
    row["point_index"] = point_index
    row["frame_id"] = frame_id
    row["camera_type"] = camera_type
    row["height"] = target_point[2]
    row["yaw"] = yaw
    row["target_x_m"] = target_point[0]
    row["target_y_m"] = target_point[1]
    row["target_z_m"] = target_point[2]
    row["target_yaw"] = yaw
    return row


def video_path_enabled(path: str | None) -> bool:
    return bool(path and path.strip() and path.strip().lower() != "none")


def map_point_to_pixel(localizer: OrbRansacLocalizer, x_m: float, y_m: float) -> tuple[int, int]:
    px = int(round(x_m / localizer.map_width_m * localizer.ref_w))
    py = int(round(y_m / localizer.map_height_m * localizer.ref_h))
    return px, py


class FlightVideoLogger:
    def __init__(
        self,
        camera_path: str | None,
        map_path: str | None,
        fps: float,
        localizer: OrbRansacLocalizer,
    ) -> None:
        self.camera_path = camera_path if video_path_enabled(camera_path) else None
        self.map_path = map_path if video_path_enabled(map_path) else None
        self.fps = fps
        self.localizer = localizer
        self.camera_writer = None
        self.map_writer = None
        self.trace: list[tuple[float, float]] = []

    def open_writer(self, path: str, size: tuple[int, int]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, self.fps, size)
        if not writer.isOpened():
            raise RuntimeError("Cannot open video writer: {}".format(path))
        return writer

    def camera_overlay(self, frame_bgr: np.ndarray, result: LocalizeResult, point_index: int, frame_id: int) -> np.ndarray:
        overlay = frame_bgr.copy()
        h, w = overlay.shape[:2]
        color = (0, 200, 0) if result.ok else (0, 0, 255)
        status = "OK" if result.ok else "FAIL"
        cv2.circle(overlay, (w // 2, h // 2), 7, color, -1)

        xy_text = "x=NA y=NA"
        if result.x_m is not None and result.y_m is not None:
            xy_text = "x={:.2f} y={:.2f}".format(result.x_m, result.y_m)
        elif result.raw_x_m is not None and result.raw_y_m is not None:
            xy_text = "raw_x={:.2f} raw_y={:.2f}".format(result.raw_x_m, result.raw_y_m)

        lines = [
            "WP {} frame {}".format(point_index, frame_id),
            "{} {}".format(status, result.message),
            xy_text,
            "matches={} inliers={} ratio={:.2f}".format(result.good_matches, result.inliers, result.inlier_ratio),
        ]
        y = 24
        for line in lines:
            cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 24
        return overlay

    def map_overlay(self, result: LocalizeResult, point_index: int) -> np.ndarray:
        canvas = cv2.cvtColor(self.localizer.reference, cv2.COLOR_GRAY2BGR)

        if result.ok and result.x_m is not None and result.y_m is not None:
            self.trace.append((result.x_m, result.y_m))

        if len(self.trace) >= 2:
            points = np.array([map_point_to_pixel(self.localizer, x, y) for x, y in self.trace], dtype=np.int32)
            cv2.polylines(canvas, [points], False, (0, 220, 220), 2)

        current_x = result.x_m if result.ok else result.raw_x_m
        current_y = result.y_m if result.ok else result.raw_y_m
        if current_x is None or current_y is None:
            last_x, last_y = self.localizer.last_smooth()
            current_x = last_x
            current_y = last_y

        color = (0, 255, 0) if result.ok else (0, 0, 255)
        if current_x is not None and current_y is not None:
            cv2.circle(canvas, map_point_to_pixel(self.localizer, current_x, current_y), 8, color, -1)

        label = "WP {} {}".format(point_index, "OK" if result.ok else "FAIL")
        cv2.putText(canvas, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(canvas, result.filter_reason or result.message, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return canvas

    def write(self, frame_bgr: np.ndarray, result: LocalizeResult, point_index: int, frame_id: int) -> None:
        if self.camera_path:
            camera_frame = self.camera_overlay(frame_bgr, result, point_index, frame_id)
            if self.camera_writer is None:
                size = (camera_frame.shape[1], camera_frame.shape[0])
                self.camera_writer = self.open_writer(self.camera_path, size)
            self.camera_writer.write(camera_frame)

        if self.map_path:
            map_frame = self.map_overlay(result, point_index)
            if self.map_writer is None:
                size = (map_frame.shape[1], map_frame.shape[0])
                self.map_writer = self.open_writer(self.map_path, size)
            self.map_writer.write(map_frame)

    def close(self) -> None:
        if self.camera_writer is not None:
            self.camera_writer.release()
        if self.map_writer is not None:
            self.map_writer.release()


def estimate_move_time(
    previous_point: tuple[float, float, float],
    next_point: tuple[float, float, float],
    speed: float,
) -> int:
    distance = math.dist(previous_point, next_point)
    return max(1, int(math.ceil(distance / speed)))


def command_local_point(drone, x: float, y: float, z: float, yaw: float, point_time: int) -> None:
    try:
        result = drone.go_to_local_point(x=x, y=y, z=z, yaw=yaw, time=point_time)
    except TypeError:
        result = drone.go_to_local_point(x, y, z, yaw, point_time)

    if result is False:
        print("WARNING: go_to_local_point returned False", file=sys.stderr)


def parse_battery_status(status) -> tuple[float, float] | None:
    if status is None:
        return None

    if isinstance(status, dict):
        voltage = status.get("voltage", status.get("battery_voltage"))
        temperature = status.get("temperature", status.get("battery_temperature"))
    elif isinstance(status, (list, tuple)) and len(status) >= 2:
        voltage, temperature = status[0], status[1]
    else:
        return None

    try:
        return float(voltage), float(temperature)
    except (TypeError, ValueError):
        return None


def check_battery_or_abort(
    drone,
    min_voltage: float,
    retries: int,
    retry_delay: float,
) -> None:
    if min_voltage <= 0:
        print("Battery voltage check disabled")
        return

    if not hasattr(drone, "get_battery_status"):
        print("WARNING: get_battery_status() unavailable")
        return

    parsed_status = None
    last_status = None
    for attempt in range(1, retries + 1):
        last_status = drone.get_battery_status()
        print("Battery status attempt {}/{}: {}".format(attempt, retries, last_status))
        parsed_status = parse_battery_status(last_status)
        if parsed_status is not None:
            break
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    if parsed_status is None:
        raise RuntimeError("Cannot read battery status: {}".format(last_status))

    voltage, temperature = parsed_status
    print("Battery voltage={:.2f} V, temperature={:.1f} C".format(voltage, temperature))

    if voltage < min_voltage:
        raise RuntimeError(
            "Battery voltage too low for flight: {:.2f} V < {:.2f} V".format(voltage, min_voltage)
        )


def wait_for_point(
    drone,
    timeout: float,
    poll_interval: float,
    stop_event: threading.Event,
) -> bool:
    if not hasattr(drone, "point_reached"):
        time.sleep(timeout)
        return True

    start = time.monotonic()
    while time.monotonic() - start < timeout and not stop_event.is_set():
        try:
            if drone.point_reached():
                return True
        except Exception as exc:
            print("WARNING: point_reached() failed: {}".format(exc), file=sys.stderr)
            break
        time.sleep(poll_interval)

    return False


def process_camera_for_seconds(
    camera,
    localizer: OrbRansacLocalizer,
    video_logger: FlightVideoLogger,
    seconds: float,
    point_index: int,
    csv_path: str | None,
    debug_dir: str | None,
    camera_type: str,
    target_point: tuple[float, float, float],
    yaw: float,
    stop_event: threading.Event,
) -> None:
    localizer.start_waypoint()
    start = time.monotonic()
    frame_id = 0

    while time.monotonic() - start < seconds and not stop_event.is_set():
        frame = camera.read()
        if frame is None:
            print("camera frame is empty", file=sys.stderr)
            time.sleep(0.05)
            continue

        result, homography, processed_frame = localizer.estimate(frame)
        row = result_row(point_index, frame_id, result, camera_type, target_point, yaw)
        print(json.dumps(row, ensure_ascii=False))
        append_csv(csv_path, row)
        video_logger.write(processed_frame, result, point_index, frame_id)

        if debug_dir:
            out = Path(debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            debug = localizer.draw_debug(processed_frame, result, homography)
            cv2.imwrite(str(out / "debug_p{:03d}_f{:05d}.jpg".format(point_index, frame_id)), debug)

        frame_id += 1
        time.sleep(0.03)


def fly_local_waypoints(args: argparse.Namespace) -> int:
    warn_show_disabled(args.show)

    localizer = OrbRansacLocalizer(
        reference_path=args.reference,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        feature=args.feature,
        nfeatures=args.nfeatures,
        ratio=args.ratio,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        ransac_threshold=args.ransac_threshold,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        reference_max_size=args.reference_max_size,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        min_inlier_ratio=args.min_inlier_ratio,
        min_homography_area_m2=args.min_homography_area_m2,
        max_homography_area_m2=args.max_homography_area_m2,
        max_position_jump=args.max_position_jump,
        ema_alpha=args.ema_alpha,
    )
    video_logger = FlightVideoLogger(
        camera_path=args.video_camera_out,
        map_path=args.video_map_out,
        fps=args.video_fps,
        localizer=localizer,
    )

    stop_event = threading.Event()
    if not args.no_command_listener:
        start_command_listener(stop_event)

    waypoints = resolve_waypoints(args)

    if args.no_flight:
        camera = OpenCvCamera(args.camera_index)
        try:
            process_camera_for_seconds(
                camera=camera,
                localizer=localizer,
                video_logger=video_logger,
                seconds=args.no_flight_seconds,
                point_index=0,
                csv_path=args.csv,
                debug_dir=args.debug_dir,
                camera_type="opencv:{}".format(args.camera_index),
                target_point=(0.0, 0.0, args.height),
                yaw=args.yaw,
                stop_event=stop_event,
            )
        finally:
            camera.close()
            video_logger.close()
        return 0

    sdk2 = import_pioneer_sdk2()
    drone = create_pioneer(sdk2)
    if args.camera_source == "pioneer-raw":
        print("WARNING: --camera-source pioneer-raw is deprecated; using SDK2 Camera.get_cv_frame().")
        camera = Sdk2Camera(sdk2, args.sdk2_camera_type, args.camera_timeout)
        camera_type = args.sdk2_camera_type.upper()
    elif args.camera_source == "sdk2":
        camera = Sdk2Camera(sdk2, args.sdk2_camera_type, args.camera_timeout)
        camera_type = args.sdk2_camera_type.upper()
    else:
        camera = OpenCvCamera(args.camera_index)
        camera_type = "opencv:{}".format(args.camera_index)

    try:
        check_battery_or_abort(
            drone=drone,
            min_voltage=args.min_battery_voltage,
            retries=args.battery_check_retries,
            retry_delay=args.battery_check_delay,
        )

        print("Arming...")
        if hasattr(drone, "arm"):
            armed = drone.arm(timeout=5, retries=1)
            if armed is False:
                raise RuntimeError("pioneer.arm() returned False")

        print("Takeoff...")
        takeoff = drone.takeoff()
        if takeoff is False:
            raise RuntimeError("pioneer.takeoff() returned False")
        time.sleep(args.takeoff_wait)

        previous_point = (0.0, 0.0, 0.0)
        for point_index, (x, y, z) in enumerate(waypoints, 1):
            if stop_event.is_set():
                print("Route interrupted before point {}. Landing...".format(point_index))
                break

            print("Point {}: x={} y={} z={}".format(point_index, x, y, z))
            next_point = (x, y, z)
            point_time = args.point_time or estimate_move_time(previous_point, next_point, args.speed)
            command_local_point(drone, x, y, z, yaw=args.yaw, point_time=point_time)

            reached = wait_for_point(
                drone=drone,
                timeout=args.move_timeout,
                poll_interval=args.poll_interval,
                stop_event=stop_event,
            )
            if not reached and not stop_event.is_set():
                print("WARNING: waypoint {} was not confirmed before timeout".format(point_index), file=sys.stderr)
            if not stop_event.is_set() and args.settle_time > 0:
                time.sleep(args.settle_time)

            process_camera_for_seconds(
                camera=camera,
                localizer=localizer,
                video_logger=video_logger,
                seconds=args.wait_per_point,
                point_index=point_index,
                csv_path=args.csv,
                debug_dir=args.debug_dir,
                camera_type=camera_type,
                target_point=next_point,
                yaw=args.yaw,
                stop_event=stop_event,
            )
            previous_point = next_point

        print("Landing...")
        drone.land()

        if hasattr(drone, "disarm"):
            print("Disarming...")
            drone.disarm()

    except KeyboardInterrupt:
        print("Interrupted. Landing...")
        if hasattr(drone, "land"):
            drone.land()
        return 130

    except Exception as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        if hasattr(drone, "land"):
            print("Landing after error...")
            try:
                drone.land()
            except Exception as land_exc:
                print("Landing failed: {}".format(land_exc), file=sys.stderr)
        return 1

    finally:
        camera.close()
        video_logger.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fly waypoints and localize by ORB + RANSAC homography.")
    parser.add_argument("--reference", default="map.jpg")
    parser.add_argument("--map-width-m", type=float, default=3.0)
    parser.add_argument("--map-height-m", type=float, default=3.0)
    parser.add_argument("--feature", choices=["orb", "akaze", "sift"], default="orb")
    parser.add_argument("--nfeatures", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--min-matches", type=int, default=18)
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--ransac-threshold", type=float, default=7.0)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--reference-max-size", type=int, default=1200)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-tile", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.55)
    parser.add_argument("--min-homography-area-m2", type=float, default=0.03)
    parser.add_argument("--max-homography-area-m2", type=float, default=0.0)
    parser.add_argument("--max-position-jump", type=float, default=0.5)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--point-time", type=int, default=6)
    parser.add_argument("--move-timeout", type=float, default=15.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--takeoff-wait", type=float, default=2.0)
    parser.add_argument("--wait-per-point", type=float, default=3.0)
    parser.add_argument("--settle-time", type=float, default=1.5)
    parser.add_argument("--camera-source", choices=["opencv", "sdk2", "pioneer-raw"], default="sdk2")
    parser.add_argument("--sdk2-camera-type", default="OPT")
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--min-battery-voltage", type=float, default=7.4)
    parser.add_argument("--battery-check-retries", type=int, default=3)
    parser.add_argument("--battery-check-delay", type=float, default=0.5)
    parser.add_argument("--csv", default="orb_ransac_localization.csv")
    parser.add_argument("--debug-dir")
    parser.add_argument("--video-camera-out", default="camera_overlay.avi")
    parser.add_argument("--video-map-out", default="map_trace.avi")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--show", action="store_true", help="Deprecated: ignored on headless/OpenCV-no-GUI builds.")
    parser.add_argument("--no-command-listener", action="store_true")
    parser.add_argument("--no-flight", action="store_true")
    parser.add_argument("--no-flight-seconds", type=float, default=20.0)
    parser.add_argument("--trajectory", choices=["waypoints", "square", "lawnmower", "cube"], default="waypoints")
    parser.add_argument("--area-size", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--high-height", type=float, default=1.5)
    parser.add_argument("--layers", type=parse_float_list)
    parser.add_argument(
        "--waypoint",
        action="append",
        type=parse_waypoint,
        default=None,
        help="Waypoint as x,y,z. Can be repeated.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.map_width_m <= 0 or args.map_height_m <= 0:
        raise ValueError("map size must be positive")
    if args.nfeatures <= 0:
        raise ValueError("--nfeatures must be positive")
    if not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be between 0 and 1")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise ValueError("--frame-width and --frame-height must be positive")
    if args.reference_max_size < 0:
        raise ValueError("--reference-max-size must be >= 0")
    if args.clahe_clip < 0:
        raise ValueError("--clahe-clip must be >= 0")
    if args.clahe_tile <= 0:
        raise ValueError("--clahe-tile must be positive")
    if not (0.0 <= args.min_inlier_ratio <= 1.0):
        raise ValueError("--min-inlier-ratio must be between 0 and 1")
    if args.min_homography_area_m2 < 0 or args.max_homography_area_m2 < 0:
        raise ValueError("--min-homography-area-m2 and --max-homography-area-m2 must be >= 0")
    if args.max_homography_area_m2 > 0 and args.max_homography_area_m2 < args.min_homography_area_m2:
        raise ValueError("--max-homography-area-m2 must be >= --min-homography-area-m2")
    if args.max_position_jump < 0:
        raise ValueError("--max-position-jump must be >= 0")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1]")
    if args.min_matches < 4:
        raise ValueError("--min-matches must be at least 4")
    if args.min_inliers < 4:
        raise ValueError("--min-inliers must be at least 4")
    if args.ransac_threshold <= 0:
        raise ValueError("--ransac-threshold must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.point_time < 0:
        raise ValueError("--point-time must be >= 0")
    if args.move_timeout <= 0 or args.poll_interval <= 0:
        raise ValueError("--move-timeout and --poll-interval must be positive")
    if args.settle_time < 0:
        raise ValueError("--settle-time must be >= 0")
    if args.camera_timeout <= 0:
        raise ValueError("--camera-timeout must be positive")
    if args.min_battery_voltage < 0:
        raise ValueError("--min-battery-voltage must be >= 0")
    if args.battery_check_retries <= 0:
        raise ValueError("--battery-check-retries must be positive")
    if args.battery_check_delay < 0:
        raise ValueError("--battery-check-delay must be >= 0")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if not args.sdk2_camera_type.strip():
        raise ValueError("--sdk2-camera-type must not be empty")
    if args.takeoff_wait < 0 or args.wait_per_point <= 0 or args.no_flight_seconds <= 0:
        raise ValueError("wait times must be valid positive values")
    if args.area_size <= 0:
        raise ValueError("--area-size must be positive")
    if args.margin < 0 or args.margin * 2 >= args.area_size:
        raise ValueError("--margin must be non-negative and smaller than half of --area-size")
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.height <= 0 or args.high_height <= 0:
        raise ValueError("--height and --high-height must be positive")
    if args.layers and any(layer <= 0 for layer in args.layers):
        raise ValueError("--layers values must be positive")
    waypoints = resolve_waypoints(args)
    for waypoint in waypoints:
        if any(not math.isfinite(value) for value in waypoint):
            raise ValueError("waypoint values must be finite numbers")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return fly_local_waypoints(args)


if __name__ == "__main__":
    raise SystemExit(main())
