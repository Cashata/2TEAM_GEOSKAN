#!/usr/bin/env python3
"""
Fly local waypoints and localize camera frames on a known 3x3 m map.

This combines two ideas:
  1. Pioneer local waypoint flight.
  2. ORB feature matching against map.jpg with homography + RANSAC.

Default behavior:
  - reference map: map.jpg
  - camera: OpenCV VideoCapture(0)
  - waypoints: a small square relative to the takeoff point

Local camera/localization test without drone:
  python fly_orb_ransac.py --no-flight --reference map.jpg --camera-index 0

Real flight:
  python3 fly_orb_ransac.py --reference map.jpg --camera-source pioneer-raw
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

LAND_COMMANDS = {"land", "stop", "q", "quit", "exit", "посадка", "сесть", "стоп"}


@dataclass
class LocalizeResult:
    ok: bool
    message: str
    x_m: float | None = None
    y_m: float | None = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0


class OrbRansacLocalizer:
    def __init__(
        self,
        reference_path: str,
        map_width_m: float,
        map_height_m: float,
        nfeatures: int,
        ratio: float,
        min_matches: int,
        min_inliers: int,
        ransac_threshold: float,
    ) -> None:
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold

        self.reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if self.reference is None:
            raise FileNotFoundError("Cannot read reference map: {}".format(reference_path))

        self.ref_h, self.ref_w = self.reference.shape[:2]
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.reference, None)
        if self.des_ref is None or len(self.kp_ref) < min_matches:
            raise RuntimeError("Reference map has too few ORB keypoints")

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def ref_pixel_to_map_m(self, px: float, py: float) -> tuple[float, float]:
        x_m = px / self.ref_w * self.map_width_m
        y_m = py / self.ref_h * self.map_height_m
        return float(x_m), float(y_m)

    def estimate(self, frame_bgr: np.ndarray) -> tuple[LocalizeResult, np.ndarray | None]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) < self.min_matches:
            return LocalizeResult(False, "too few frame keypoints"), None

        knn = self.matcher.knnMatch(des_frame, self.des_ref, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            return LocalizeResult(False, "too few good matches", good_matches=len(good)), None

        pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([self.kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            pts_frame,
            pts_ref,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if homography is None or mask is None:
            return LocalizeResult(False, "RANSAC homography failed", good_matches=len(good)), None

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / max(len(good), 1)
        if inliers < self.min_inliers:
            return (
                LocalizeResult(
                    False,
                    "too few RANSAC inliers",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                ),
                homography,
            )

        h, w = gray.shape[:2]
        center_frame = np.float32([[[w / 2.0, h / 2.0]]])
        center_ref = cv2.perspectiveTransform(center_frame, homography)[0, 0]
        ref_x, ref_y = float(center_ref[0]), float(center_ref[1])

        if not (0.0 <= ref_x <= self.ref_w and 0.0 <= ref_y <= self.ref_h):
            return (
                LocalizeResult(
                    False,
                    "estimated center is outside reference map",
                    good_matches=len(good),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                ),
                homography,
            )

        x_m, y_m = self.ref_pixel_to_map_m(ref_x, ref_y)
        return (
            LocalizeResult(
                True,
                "ok",
                x_m=x_m,
                y_m=y_m,
                good_matches=len(good),
                inliers=inliers,
                inlier_ratio=inlier_ratio,
            ),
            homography,
        )

    def draw_debug(self, frame_bgr: np.ndarray, result: LocalizeResult, homography: np.ndarray | None) -> np.ndarray:
        ref_debug = cv2.cvtColor(self.reference, cv2.COLOR_GRAY2BGR)

        if homography is not None:
            h, w = frame_bgr.shape[:2]
            corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
            projected = cv2.perspectiveTransform(corners, homography)
            cv2.polylines(ref_debug, [np.int32(projected[0])], True, (255, 0, 0), 2)

        if result.ok and result.x_m is not None and result.y_m is not None:
            px = int(round(result.x_m / self.map_width_m * self.ref_w))
            py = int(round(result.y_m / self.map_height_m * self.ref_h))
            cv2.circle(ref_debug, (px, py), 8, (0, 255, 0), -1)
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


class PioneerRawCamera:
    def __init__(self, drone) -> None:
        self.drone = drone

    def read(self) -> np.ndarray | None:
        raw = self.drone.get_raw_video_frame()
        if raw is None:
            return None
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    def close(self) -> None:
        pass


def import_pioneer_class():
    try:
        module = importlib.import_module("pioneer_sdk")
    except ModuleNotFoundError as exc:
        raise RuntimeError("pioneer_sdk is not installed in this Python environment") from exc
    return module.Pioneer


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
        print("Type 'land', 'stop', 'q', 'посадка' or 'сесть' + Enter for graceful landing.")
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


def result_row(point_index: int, result: LocalizeResult) -> dict[str, object]:
    row = asdict(result)
    row["timestamp"] = time.time()
    row["point_index"] = point_index
    return row


def command_local_point(drone, x: float, y: float, z: float, speed: float, yaw: float) -> None:
    if hasattr(drone, "move_to_local"):
        drone.move_to_local(x, y, z, speed=speed)
        return

    if hasattr(drone, "go_to_local_point"):
        try:
            drone.go_to_local_point(x=x, y=y, z=z, yaw=yaw)
        except TypeError:
            drone.go_to_local_point(x, y, z, yaw)
        return

    raise RuntimeError("Drone object has neither move_to_local() nor go_to_local_point()")


def process_camera_for_seconds(
    camera,
    localizer: OrbRansacLocalizer,
    seconds: float,
    point_index: int,
    csv_path: str | None,
    show: bool,
    debug_dir: str | None,
    stop_event: threading.Event,
) -> None:
    start = time.monotonic()
    frame_id = 0

    while time.monotonic() - start < seconds and not stop_event.is_set():
        frame = camera.read()
        if frame is None:
            print("camera frame is empty", file=sys.stderr)
            time.sleep(0.05)
            continue

        result, homography = localizer.estimate(frame)
        row = result_row(point_index, result)
        row["frame_id"] = frame_id
        print(json.dumps(row, ensure_ascii=False))
        append_csv(csv_path, row)

        if debug_dir:
            out = Path(debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            debug = localizer.draw_debug(frame, result, homography)
            cv2.imwrite(str(out / "debug_p{:03d}_f{:05d}.jpg".format(point_index, frame_id)), debug)

        if show:
            debug = localizer.draw_debug(frame, result, homography)
            cv2.imshow("frame", frame)
            cv2.imshow("map_debug", debug)
            if cv2.waitKey(1) == 27:
                print("ESC pressed, stopping camera loop")
                stop_event.set()
                break

        frame_id += 1
        time.sleep(0.03)


def fly_local_waypoints(args: argparse.Namespace) -> int:
    localizer = OrbRansacLocalizer(
        reference_path=args.reference,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        nfeatures=args.nfeatures,
        ratio=args.ratio,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        ransac_threshold=args.ransac_threshold,
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
                seconds=args.no_flight_seconds,
                point_index=0,
                csv_path=args.csv,
                show=args.show,
                debug_dir=args.debug_dir,
                stop_event=stop_event,
            )
        finally:
            camera.close()
            cv2.destroyAllWindows()
        return 0

    Pioneer = import_pioneer_class()
    drone = Pioneer()
    camera = PioneerRawCamera(drone) if args.camera_source == "pioneer-raw" else OpenCvCamera(args.camera_index)

    try:
        print("Arming...")
        if hasattr(drone, "arm"):
            drone.arm()

        print("Takeoff...")
        drone.takeoff()
        time.sleep(args.takeoff_wait)

        for point_index, (x, y, z) in enumerate(waypoints, 1):
            if stop_event.is_set():
                print("Route interrupted before point {}. Landing...".format(point_index))
                break

            print("Point {}: x={} y={} z={}".format(point_index, x, y, z))
            command_local_point(drone, x, y, z, speed=args.speed, yaw=args.yaw)

            process_camera_for_seconds(
                camera=camera,
                localizer=localizer,
                seconds=args.wait_per_point,
                point_index=point_index,
                csv_path=args.csv,
                show=args.show,
                debug_dir=args.debug_dir,
                stop_event=stop_event,
            )

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
                if hasattr(drone, "emergency_stop"):
                    print("Emergency stop...")
                    drone.emergency_stop()
        elif hasattr(drone, "emergency_stop"):
            print("Emergency stop...")
            drone.emergency_stop()
        return 1

    finally:
        camera.close()
        if hasattr(drone, "close"):
            drone.close()
        cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fly waypoints and localize by ORB + RANSAC homography.")
    parser.add_argument("--reference", default="map.jpg")
    parser.add_argument("--map-width-m", type=float, default=3.0)
    parser.add_argument("--map-height-m", type=float, default=3.0)
    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--takeoff-wait", type=float, default=2.0)
    parser.add_argument("--wait-per-point", type=float, default=3.0)
    parser.add_argument("--camera-source", choices=["opencv", "pioneer-raw"], default="pioneer-raw")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--csv", default="orb_ransac_localization.csv")
    parser.add_argument("--debug-dir")
    parser.add_argument("--show", action="store_true")
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
    if args.min_matches < 4:
        raise ValueError("--min-matches must be at least 4")
    if args.min_inliers < 4:
        raise ValueError("--min-inliers must be at least 4")
    if args.ransac_threshold <= 0:
        raise ValueError("--ransac-threshold must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
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
