#!/usr/bin/env python3
"""Manual WASD flight helper for Pioneer Mini 2 / Pioneer-SDK2."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geoscan_mission.flight.camera import Sdk2Camera
from geoscan_mission.flight.control import (
    check_battery_or_abort,
    create_pioneer,
    import_pioneer_sdk2,
    return_to_launch_or_land,
)


def send_manual_speed(drone, vx: float, vy: float, vz: float, yaw_rate: float, interval: float) -> None:
    try:
        result = drone.set_manual_speed_body_fixed(
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            interval=interval,
        )
    except TypeError:
        result = drone.set_manual_speed_body_fixed(vx, vy, vz, yaw_rate)

    if result is False:
        raise RuntimeError("set_manual_speed_body_fixed() returned False")


def safe_hold(drone, interval: float) -> None:
    if hasattr(drone, "set_manual_speed_body_fixed"):
        send_manual_speed(drone, 0.0, 0.0, 0.0, 0.0, interval)


def safe_land(drone, interval: float) -> None:
    safe_hold(drone, interval)
    if hasattr(drone, "land"):
        drone.land()


def arm_drone(drone) -> None:
    if hasattr(drone, "arm"):
        result = drone.arm(timeout=5, retries=1)
        if result is False:
            raise RuntimeError("pioneer.arm() returned False")


def takeoff_drone(drone) -> None:
    if hasattr(drone, "takeoff"):
        result = drone.takeoff()
        if result is False:
            raise RuntimeError("pioneer.takeoff() returned False")


def movement_for_key(key: int, args: argparse.Namespace) -> tuple[float, float, float, float, str] | None:
    bindings = {
        ord("w"): (args.speed, 0.0, 0.0, 0.0, "forward"),
        ord("s"): (-args.speed, 0.0, 0.0, 0.0, "back"),
        ord("a"): (0.0, args.speed, 0.0, 0.0, "left"),
        ord("d"): (0.0, -args.speed, 0.0, 0.0, "right"),
        ord("i"): (0.0, 0.0, args.vertical_speed, 0.0, "up"),
        ord("k"): (0.0, 0.0, -args.vertical_speed, 0.0, "down"),
        ord("q"): (0.0, 0.0, 0.0, args.yaw_rate, "yaw left"),
        ord("e"): (0.0, 0.0, 0.0, -args.yaw_rate, "yaw right"),
    }
    return bindings.get(key)


def draw_hud(cv2, np, frame, status: str, command: str, args: argparse.Namespace):
    if frame is None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

    lines = [
        "WASD manual control",
        "1 arm  2 disarm  3 takeoff  4 land  R rtl  Esc/X land+exit",
        "W/S forward/back  A/D left/right  I/K up/down  Q/E yaw",
        "Space: hold stop",
        "status: {}".format(status),
        "command: {}".format(command),
        "speed xy={:.2f} z={:.2f} yaw_rate={:.2f}".format(args.speed, args.vertical_speed, args.yaw_rate),
    ]
    y = 28
    for index, line in enumerate(lines):
        color = (0, 255, 0) if index >= 4 else (255, 255, 255)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 26
    return frame


def read_camera_frame(camera):
    if camera is None:
        return None
    try:
        return camera.read()
    except Exception as exc:
        print("WARNING: camera read failed: {}".format(exc), file=sys.stderr)
        return None


def resolve_video_path(args: argparse.Namespace) -> Path | None:
    if args.no_video:
        return None
    raw_path = args.video_out.strip()
    if not raw_path or raw_path.lower() == "none":
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = Path(raw_path.format(timestamp=timestamp)).expanduser()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    return video_path


def create_video_writer(cv2, video_path: Path, frame, fps: float):
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open video writer: {}".format(video_path))
    return writer


def run_control(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    sdk2 = import_pioneer_sdk2()
    drone = create_pioneer(sdk2)
    if not hasattr(drone, "set_manual_speed_body_fixed"):
        raise RuntimeError("WASD control requires pioneer.set_manual_speed_body_fixed()")

    camera = None
    if not args.no_camera:
        try:
            camera = Sdk2Camera(sdk2, args.sdk2_camera_type, args.camera_timeout)
        except Exception as exc:
            print("WARNING: SDK2 camera unavailable, using a blank control window: {}".format(exc), file=sys.stderr)

    check_battery_or_abort(
        drone=drone,
        min_voltage=args.min_battery_voltage,
        retries=args.battery_check_retries,
        retry_delay=args.battery_check_delay,
    )

    is_armed = False
    flight_started = False
    status = "ready"
    command = "hold"
    video_path = resolve_video_path(args)
    video_writer = None

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        if args.takeoff_on_start:
            arm_drone(drone)
            is_armed = True
            takeoff_drone(drone)
            flight_started = True
            status = "airborne"
            time.sleep(args.takeoff_wait)

        while True:
            frame = read_camera_frame(camera)
            frame = draw_hud(cv2, np, frame, status, command, args)
            if video_path is not None:
                if video_writer is None:
                    video_writer = create_video_writer(cv2, video_path, frame, args.video_fps)
                    print("Recording WASD video to {}".format(video_path))
                video_writer.write(frame)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(max(1, int(args.loop_interval * 1000)))
            if key != -1:
                key &= 0xFF

            command = "hold"
            if key == -1:
                if flight_started:
                    safe_hold(drone, args.command_interval)
                continue

            if key in (27, ord("x")):
                status = "landing"
                command = "land+exit"
                safe_land(drone, args.command_interval)
                flight_started = False
                break
            if key == ord(" "):
                safe_hold(drone, args.command_interval)
                command = "hold stop"
                continue
            if key == ord("1"):
                arm_drone(drone)
                is_armed = True
                status = "armed"
                command = "arm"
                continue
            if key == ord("2"):
                safe_hold(drone, args.command_interval)
                if hasattr(drone, "disarm"):
                    drone.disarm()
                is_armed = False
                flight_started = False
                status = "disarmed"
                command = "disarm"
                continue
            if key == ord("3"):
                if not is_armed:
                    arm_drone(drone)
                    is_armed = True
                takeoff_drone(drone)
                flight_started = True
                status = "airborne"
                command = "takeoff"
                time.sleep(args.takeoff_wait)
                continue
            if key == ord("4"):
                status = "landing"
                command = "land"
                safe_land(drone, args.command_interval)
                flight_started = False
                continue
            if key == ord("r"):
                status = "rtl"
                command = "rtl"
                return_to_launch_or_land(
                    drone=drone,
                    height=args.rtl_height,
                    yaw=0.0,
                    point_time=args.point_time,
                    timeout=args.move_timeout,
                    poll_interval=args.poll_interval,
                )
                flight_started = False
                break

            movement = movement_for_key(key, args)
            if movement is None:
                if flight_started:
                    safe_hold(drone, args.command_interval)
                command = "unknown key"
                continue

            vx, vy, vz, yaw_rate, command = movement
            if flight_started:
                send_manual_speed(drone, vx, vy, vz, yaw_rate, args.command_interval)
            else:
                command = "{} ignored before takeoff".format(command)

    except KeyboardInterrupt:
        print("Interrupted. Landing...", file=sys.stderr)
        if flight_started:
            safe_land(drone, args.command_interval)
            flight_started = False
        return 130
    finally:
        if flight_started:
            try:
                safe_land(drone, args.command_interval)
            except Exception as exc:
                print("WARNING: landing on exit failed: {}".format(exc), file=sys.stderr)
        if camera is not None:
            camera.close()
        if video_writer is not None:
            video_writer.release()
            print("Saved WASD video: {}".format(video_path))
        cv2.destroyAllWindows()
        if hasattr(drone, "close_connection"):
            drone.close_connection()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.15, help="Body-fixed XY speed in m/s.")
    parser.add_argument("--vertical-speed", type=float, default=0.12, help="Body-fixed Z speed in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.4, help="Yaw rate in rad/s.")
    parser.add_argument("--command-interval", type=float, default=0.2)
    parser.add_argument("--loop-interval", type=float, default=0.03)
    parser.add_argument("--takeoff-on-start", action="store_true")
    parser.add_argument("--takeoff-wait", type=float, default=2.0)
    parser.add_argument("--rtl-height", type=float, default=1.0)
    parser.add_argument("--point-time", type=int, default=4)
    parser.add_argument("--move-timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--min-battery-voltage", type=float, default=7.4)
    parser.add_argument("--battery-check-retries", type=int, default=3)
    parser.add_argument("--battery-check-delay", type=float, default=0.5)
    parser.add_argument("--sdk2-camera-type", default="OPT")
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--video-out", default="wasd_flight_{timestamp}.avi")
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--window-name", default="pioneer_wasd_control")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.speed <= 0 or args.vertical_speed <= 0 or args.yaw_rate <= 0:
        raise ValueError("--speed, --vertical-speed and --yaw-rate must be positive")
    if args.command_interval <= 0 or args.loop_interval <= 0:
        raise ValueError("--command-interval and --loop-interval must be positive")
    if args.rtl_height <= 0:
        raise ValueError("--rtl-height must be positive")
    if args.point_time < 0:
        raise ValueError("--point-time must be >= 0")
    if args.move_timeout <= 0 or args.poll_interval <= 0:
        raise ValueError("--move-timeout and --poll-interval must be positive")
    if args.camera_timeout <= 0:
        raise ValueError("--camera-timeout must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.battery_check_retries <= 0:
        raise ValueError("--battery-check-retries must be positive")
    if args.battery_check_delay < 0:
        raise ValueError("--battery-check-delay must be >= 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return run_control(args)


if __name__ == "__main__":
    raise SystemExit(main())
