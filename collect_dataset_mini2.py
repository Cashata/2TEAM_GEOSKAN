#!/usr/bin/env python3
"""
Collect a visual dataset over a 3x3 m field with Geoscan Pioneer Mini 2.

The script flies a layered "cube" route over the field: several XY grid
points at several heights and yaw angles. At every pose it saves camera
frames as JPG files and writes metadata for each saved image.

Run on Pioneer Mini 2 / Pioneer OS, where Pioneer-SDK2 is available.

Safe default:
  origin=center means the takeoff point is the center of the 3x3 m field.
  With area=3.0 and margin=0.25, XY points stay inside [-1.25, 1.25].

Example:
  python3 collect_dataset_mini2.py

Dry route preview without importing drone/camera SDK:
  python3 collect_dataset_mini2.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = "/home/geoscan/dataset_landmarks"
DEFAULT_HEIGHTS = "0.6,0.8,1.0,1.2"
DEFAULT_YAWS = "0,45,90,135,180,225,270,315"


def import_pioneer_sdk2():
    """Import Pioneer-SDK2 only when a real flight is started.

    Pioneer-SDK2 is already available in Pioneer OS on Mini 2. Installing it
    into a Windows/PyCharm interpreter is optional for local editing and often
    fails while building native dependencies. Use --dry-run locally instead.
    """
    try:
        return importlib.import_module("pioneer_sdk2")
    except ModuleNotFoundError as exc:
        if exc.name != "pioneer_sdk2":
            raise
        raise RuntimeError(
            "pioneer_sdk2 is not installed in this Python environment. "
            "Run this script on Pioneer Mini 2 / Pioneer OS, where SDK2 is "
            "preinstalled, or use --dry-run on a local PC to preview the route."
        ) from exc


@dataclass(frozen=True)
class Pose:
    index: int
    layer_index: int
    grid_index: int
    x: float
    y: float
    z: float
    yaw: float


def parse_float_list(value: str) -> list[float]:
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            items.append(float(raw))
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one number")
    return items


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def build_grid(area_size: float, margin: float, grid_size: int, origin: str) -> list[tuple[float, float]]:
    if area_size <= 0:
        raise ValueError("area_size must be positive")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    usable = area_size - 2.0 * margin
    if usable <= 0:
        raise ValueError("margin is too large for selected area_size")

    if origin == "center":
        lo = -area_size / 2.0 + margin
        hi = area_size / 2.0 - margin
    elif origin == "corner":
        lo = margin
        hi = area_size - margin
    else:
        raise ValueError("origin must be 'center' or 'corner'")

    xs = linspace(lo, hi, grid_size)
    ys = linspace(lo, hi, grid_size)

    # Lawn-mower order: no long return jumps at the end of each row.
    points: list[tuple[float, float]] = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((round(x, 3), round(y, 3)))
    return points


def build_route(
    grid_points: list[tuple[float, float]],
    heights: Iterable[float],
    yaws: Iterable[float],
) -> list[Pose]:
    route: list[Pose] = []
    idx = 0

    for layer_index, z in enumerate(heights):
        # Reverse the grid every second layer to reduce repositioning.
        layer_grid = grid_points if layer_index % 2 == 0 else list(reversed(grid_points))

        for grid_index, (x, y) in enumerate(layer_grid):
            for yaw in yaws:
                route.append(
                    Pose(
                        index=idx,
                        layer_index=layer_index,
                        grid_index=grid_index,
                        x=float(x),
                        y=float(y),
                        z=float(z),
                        yaw=float(yaw),
                    )
                )
                idx += 1

    return route


def make_session_dir(base_dir: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base_dir).expanduser() / f"session_{stamp}"
    (session_dir / "images_raw").mkdir(parents=True, exist_ok=False)
    return session_dir


def write_config(session_dir: Path, args: argparse.Namespace, route: list[Pose]) -> None:
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "route_size": len(route),
        "route": [asdict(pose) for pose in route],
    }
    (session_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_route_preview(session_dir: Path, route: list[Pose]) -> None:
    with (session_dir / "route_preview.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "layer_index", "grid_index", "x", "y", "z", "yaw"],
        )
        writer.writeheader()
        for pose in route:
            writer.writerow(asdict(pose))


def append_metadata(metadata_path: Path, row: dict[str, object]) -> None:
    file_exists = metadata_path.exists()
    fieldnames = [
        "timestamp",
        "image_path",
        "pose_index",
        "layer_index",
        "grid_index",
        "x",
        "y",
        "z",
        "yaw",
        "frame_index",
    ]

    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def wait_for_point(pioneer, timeout: float, poll_interval: float) -> bool:
    """Wait until the SDK reports waypoint arrival.

    SDK examples expose point_reached(). If it is absent on a particular SDK
    build, fall back to a fixed wait equal to timeout.
    """
    if not hasattr(pioneer, "point_reached"):
        time.sleep(timeout)
        return True

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if pioneer.point_reached():
                return True
        except Exception as exc:
            print(f"WARNING: point_reached() failed: {exc}", file=sys.stderr)
            break
        time.sleep(poll_interval)

    return False


def save_frames(cv2, camera, session_dir: Path, pose: Pose, frames_per_pose: int, frame_interval: float) -> int:
    images_dir = session_dir / "images_raw"
    metadata_path = session_dir / "metadata.csv"
    saved = 0

    for frame_index in range(frames_per_pose):
        frame = camera.get_cv_frame(timeout=2.0)
        if frame is None:
            print(f"WARNING: empty frame at pose {pose.index}, frame {frame_index}", file=sys.stderr)
            time.sleep(frame_interval)
            continue

        filename = (
            f"pose_{pose.index:05d}"
            f"_z{pose.z:.2f}"
            f"_x{pose.x:.2f}"
            f"_y{pose.y:.2f}"
            f"_yaw{pose.yaw:.0f}"
            f"_f{frame_index:02d}.jpg"
        )
        image_path = images_dir / filename
        ok = cv2.imwrite(str(image_path), frame)
        if not ok:
            print(f"WARNING: failed to save {image_path}", file=sys.stderr)
            time.sleep(frame_interval)
            continue

        append_metadata(
            metadata_path,
            {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "image_path": str(image_path.relative_to(session_dir)),
                "pose_index": pose.index,
                "layer_index": pose.layer_index,
                "grid_index": pose.grid_index,
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "yaw": pose.yaw,
                "frame_index": frame_index,
            },
        )
        saved += 1
        time.sleep(frame_interval)

    return saved


def create_pioneer():
    sdk2 = import_pioneer_sdk2()

    # Defaults are wait_callback=True and safety_command=True in SDK2 docs.
    return sdk2.Pioneer(wait_callback=True, safety_command=True)


def create_camera():
    sdk2 = import_pioneer_sdk2()

    try:
        return sdk2.Camera(camera_type=sdk2.CameraType.MAIN)
    except TypeError:
        # Some examples initialize Camera with a positional camera type.
        return sdk2.Camera(sdk2.CameraType.MAIN)


def set_camera_angle(angle: float) -> None:
    try:
        sdk2 = import_pioneer_sdk2()

        servo = sdk2.ServoCamera()
        servo.set_angle(angle)
        print(f"Camera servo angle set to {angle} degrees")
    except Exception as exc:
        print(f"WARNING: camera servo is unavailable or not configured: {exc}", file=sys.stderr)


def run_mission(args: argparse.Namespace) -> int:
    grid = build_grid(args.area_size, args.margin, args.grid_size, args.origin)
    route = build_route(grid, args.heights, args.yaws)
    session_dir = make_session_dir(args.out_dir)
    write_config(session_dir, args, route)
    write_route_preview(session_dir, route)

    print(f"Dataset session: {session_dir}")
    print(f"Route poses: {len(route)}")
    print(f"Expected images: {len(route) * args.frames_per_pose}")

    if args.dry_run:
        print("Dry run only. Route preview and config were written.")
        return 0

    import cv2

    pioneer = None
    camera = None
    total_saved = 0

    try:
        pioneer = create_pioneer()
        camera = create_camera()
        set_camera_angle(args.camera_angle)

        print("Arming...")
        if not pioneer.arm(timeout=5, retries=1):
            raise RuntimeError("pioneer.arm() returned False")

        print("Takeoff...")
        if not pioneer.takeoff():
            raise RuntimeError("pioneer.takeoff() returned False")
        time.sleep(args.takeoff_settle)

        for pose in route:
            print(
                f"Pose {pose.index + 1}/{len(route)}: "
                f"x={pose.x:.2f}, y={pose.y:.2f}, z={pose.z:.2f}, yaw={pose.yaw:.0f}"
            )
            ok = pioneer.go_to_local_point(
                x=pose.x,
                y=pose.y,
                z=pose.z,
                yaw=pose.yaw,
                time=args.point_time,
            )
            if ok is False:
                print(f"WARNING: go_to_local_point returned False at pose {pose.index}", file=sys.stderr)

            reached = wait_for_point(pioneer, args.move_timeout, args.poll_interval)
            if not reached:
                print(f"WARNING: waypoint timeout at pose {pose.index}", file=sys.stderr)
            time.sleep(args.settle_time)

            total_saved += save_frames(
                cv2=cv2,
                camera=camera,
                session_dir=session_dir,
                pose=pose,
                frames_per_pose=args.frames_per_pose,
                frame_interval=args.frame_interval,
            )

        print("Mission complete. Landing...")
        pioneer.land()

    except KeyboardInterrupt:
        print("Interrupted by user. Landing...")
        if pioneer is not None:
            pioneer.land()
        return 130

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if pioneer is not None:
            try:
                pioneer.land()
            except Exception as land_exc:
                print(f"ERROR during landing: {land_exc}", file=sys.stderr)
        return 1

    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception as exc:
                print(f"WARNING: camera.stop() failed: {exc}", file=sys.stderr)

    print(f"Saved images: {total_saved}")
    print(f"Metadata: {session_dir / 'metadata.csv'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Mini 2 camera frames over a 3x3 m field.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--area-size", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--origin", choices=["center", "corner"], default="center")
    parser.add_argument("--heights", type=parse_float_list, default=parse_float_list(DEFAULT_HEIGHTS))
    parser.add_argument("--yaws", type=parse_float_list, default=parse_float_list(DEFAULT_YAWS))
    parser.add_argument("--frames-per-pose", type=int, default=3)
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--point-time", type=int, default=0)
    parser.add_argument("--move-timeout", type=float, default=15.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--takeoff-settle", type=float, default=3.0)
    parser.add_argument("--camera-angle", type=float, default=-90.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create config/route files. Do not import SDK or fly.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.frames_per_pose <= 0:
        raise ValueError("--frames-per-pose must be positive")
    if args.frame_interval < 0:
        raise ValueError("--frame-interval must be >= 0")
    if args.move_timeout <= 0:
        raise ValueError("--move-timeout must be positive")
    if args.camera_angle < -90 or args.camera_angle > 30:
        raise ValueError("--camera-angle must be in [-90, 30] degrees")

    for height in args.heights:
        if height <= 0:
            raise ValueError("all heights must be positive")
    for yaw in args.yaws:
        if not math.isfinite(yaw):
            raise ValueError("all yaws must be finite numbers")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    return run_mission(args)


if __name__ == "__main__":
    raise SystemExit(main())
