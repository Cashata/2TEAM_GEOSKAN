#!/usr/bin/env python3
"""CLI for waypoint flight with ORB/RANSAC localization and optional ArUco logging."""

from __future__ import annotations

import argparse
import math
import sys
import threading

from geoscan_mission.flight.camera import OpenCvCamera, Sdk2Camera
from geoscan_mission.flight.control import (
    check_battery_or_abort,
    command_local_point,
    create_pioneer,
    estimate_move_time,
    import_pioneer_sdk2,
    sleep_while_recording,
    start_command_listener,
    wait_for_point,
    warn_show_disabled,
)
from geoscan_mission.recording import ContinuousFlightRecorder, FlightVideoLogger
from geoscan_mission.trajectory.patterns import parse_float_list, parse_waypoint, resolve_waypoints
from geoscan_mission.vision.aruco import ArucoDetector, DEFAULT_DICTIONARY
from geoscan_mission.vision.localization import OrbRansacLocalizer


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
    aruco_detector = ArucoDetector(dictionary_name=args.aruco_dict) if args.aruco else None

    stop_event = threading.Event()
    if not args.no_command_listener:
        start_command_listener(stop_event)

    waypoints = resolve_waypoints(args)

    if args.no_flight:
        camera = OpenCvCamera(args.camera_index)
        recorder = ContinuousFlightRecorder(
            camera=camera,
            localizer=localizer,
            video_logger=video_logger,
            csv_path=args.csv,
            debug_dir=args.debug_dir,
            camera_type="opencv:{}".format(args.camera_index),
            yaw=args.yaw,
            stop_event=stop_event,
            aruco_detector=aruco_detector,
        )
        recorder.set_context(0, (0.0, 0.0, args.height), args.yaw, "no_flight", reset_tracking=True)
        recorder.start()
        try:
            sleep_while_recording(args.no_flight_seconds, recorder, stop_event)
            recorder.raise_if_failed()
        finally:
            recorder.stop()
            recorder.join()
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

    recorder = ContinuousFlightRecorder(
        camera=camera,
        localizer=localizer,
        video_logger=video_logger,
        csv_path=args.csv,
        debug_dir=args.debug_dir,
        camera_type=camera_type,
        yaw=args.yaw,
        stop_event=stop_event,
        aruco_detector=aruco_detector,
    )
    recorder.set_context(0, (0.0, 0.0, 0.0), args.yaw, "preflight", reset_tracking=True)
    recorder.start()

    is_armed = False
    flight_started = False

    try:
        check_battery_or_abort(
            drone=drone,
            min_voltage=args.min_battery_voltage,
            retries=args.battery_check_retries,
            retry_delay=args.battery_check_delay,
        )
        recorder.raise_if_failed()

        recorder.set_context(0, (0.0, 0.0, 0.0), args.yaw, "arm")
        print("Arming...")
        if hasattr(drone, "arm"):
            armed = drone.arm(timeout=5, retries=1)
            if armed is False:
                raise RuntimeError("pioneer.arm() returned False")
            is_armed = True
        recorder.raise_if_failed()

        recorder.set_context(0, (0.0, 0.0, args.height), args.yaw, "takeoff", reset_tracking=True)
        print("Takeoff...")
        takeoff = drone.takeoff()
        if takeoff is False:
            raise RuntimeError("pioneer.takeoff() returned False")
        flight_started = True
        recorder.set_context(0, (0.0, 0.0, args.height), args.yaw, "takeoff_wait")
        sleep_while_recording(args.takeoff_wait, recorder, stop_event)

        previous_point = (0.0, 0.0, 0.0)
        for point_index, (x, y, z) in enumerate(waypoints, 1):
            if stop_event.is_set():
                print("Route interrupted before point {}. Landing...".format(point_index))
                break

            print("Point {}: x={} y={} z={}".format(point_index, x, y, z))
            next_point = (x, y, z)
            point_time = args.point_time or estimate_move_time(previous_point, next_point, args.speed)
            recorder.set_context(point_index, next_point, args.yaw, "move", reset_tracking=True)
            command_local_point(drone, x, y, z, yaw=args.yaw, point_time=point_time)

            reached = wait_for_point(
                drone=drone,
                timeout=args.move_timeout,
                poll_interval=args.poll_interval,
                stop_event=stop_event,
            )
            recorder.raise_if_failed()
            if not reached and not stop_event.is_set():
                print("WARNING: waypoint {} was not confirmed before timeout".format(point_index), file=sys.stderr)
            if not stop_event.is_set() and args.settle_time > 0:
                recorder.set_context(point_index, next_point, args.yaw, "settle")
                sleep_while_recording(args.settle_time, recorder, stop_event)

            recorder.set_context(point_index, next_point, args.yaw, "hold")
            sleep_while_recording(args.wait_per_point, recorder, stop_event)
            previous_point = next_point

        recorder.set_context(len(waypoints) + 1, previous_point, args.yaw, "landing")
        print("Landing...")
        drone.land()
        flight_started = False
        if args.landing_record_time > 0:
            recorder.set_context(len(waypoints) + 1, previous_point, args.yaw, "landed")
            sleep_while_recording(args.landing_record_time, recorder, stop_event)

        if hasattr(drone, "disarm"):
            recorder.set_context(len(waypoints) + 1, previous_point, args.yaw, "disarm")
            print("Disarming...")
            drone.disarm()

    except KeyboardInterrupt:
        print("Interrupted. Landing...")
        if flight_started and hasattr(drone, "land"):
            recorder.set_context(0, (0.0, 0.0, 0.0), args.yaw, "interrupt_landing")
            drone.land()
        elif is_armed and hasattr(drone, "disarm"):
            recorder.set_context(0, (0.0, 0.0, 0.0), args.yaw, "interrupt_disarm")
            drone.disarm()
        return 130

    except Exception as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        if flight_started and hasattr(drone, "land"):
            print("Landing after error...")
            try:
                recorder.set_context(0, (0.0, 0.0, 0.0), args.yaw, "error_landing")
                drone.land()
            except Exception as land_exc:
                print("Landing failed: {}".format(land_exc), file=sys.stderr)
        elif is_armed and hasattr(drone, "disarm"):
            print("Disarming after preflight/takeoff error...")
            recorder.set_context(0, (0.0, 0.0, 0.0), args.yaw, "error_disarm")
            drone.disarm()
        return 1

    finally:
        recorder.stop()
        recorder.join()
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
    parser.add_argument("--landing-record-time", type=float, default=3.0)
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
    parser.add_argument("--aruco", action="store_true", help="Detect mission ArUco targets and add them to logs/overlay.")
    parser.add_argument("--aruco-dict", default=DEFAULT_DICTIONARY, help="OpenCV ArUco dictionary name.")
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
    if args.landing_record_time < 0:
        raise ValueError("--landing-record-time must be >= 0")
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
    if args.aruco and not args.aruco_dict.strip():
        raise ValueError("--aruco-dict must not be empty")
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
