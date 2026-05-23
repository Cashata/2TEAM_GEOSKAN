#!/usr/bin/env python3
"""Autonomous ID15 pickup demo.

Mission sequence:
1. Take off and climb to the requested height.
2. Lock current ORB map coordinates to current Pioneer LPS coordinates.
3. Fly a small local scan pattern while the camera searches for ArUco ID 15.
4. Convert the detected ID15 map coordinate to local LPS coordinates.
5. Fly above ID15, land, take off again, return to the launch point, and land.

The script also serves a live MJPEG preview with ORB/ArUco overlays.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

import aruco_hand_check
import aruco_tracker_standalone as aruco
from geoscan_mission.flight.camera import OpenCvCamera, Sdk2Camera, UndistortedCamera
from geoscan_mission.flight.control import (
    FlightCommandState,
    check_battery_or_abort,
    command_local_point,
    create_pioneer,
    estimate_move_time,
    import_pioneer_sdk2,
    start_command_listener,
)
from geoscan_mission.recording import FlightEventLogger, append_csv
from geoscan_mission.vision.calibration import load_camera_calibration
from geoscan_mission.vision.localization import LocalizeResult, OrbRansacLocalizer


Point3 = tuple[float, float, float]


class MissionState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.phase = "setup"
        self.target_point: Point3 | None = None
        self.frame_id = 0
        self.last_result: LocalizeResult | None = None
        self.last_visible_ids: list[int] = []
        self.anchor_map_xy: tuple[float, float] | None = None
        self.anchor_local_xy: tuple[float, float] | None = None
        self.map_to_local_matrix: np.ndarray | None = None
        self.id15_record: aruco.MarkerRecord | None = None
        self.id15_map_xy: tuple[float, float] | None = None
        self.id15_local_xy: tuple[float, float] | None = None
        self.id15_seen_time: float | None = None
        self.latest_error: str | None = None

    def set_phase(self, phase: str, target_point: Point3 | None = None) -> None:
        with self.lock:
            self.phase = phase
            self.target_point = target_point

    def set_orb_result(self, result: LocalizeResult, visible_ids: list[int]) -> None:
        with self.lock:
            self.last_result = result
            self.last_visible_ids = list(visible_ids)
            self.frame_id += 1

    def set_anchor(self, map_xy: tuple[float, float], local_xy: tuple[float, float]) -> None:
        with self.lock:
            self.anchor_map_xy = map_xy
            self.anchor_local_xy = local_xy

    def set_map_to_local_matrix(self, matrix: np.ndarray) -> None:
        with self.lock:
            self.map_to_local_matrix = np.asarray(matrix, dtype=np.float64)

    def map_to_local_xy(self, map_xy: tuple[float, float]) -> tuple[float, float] | None:
        with self.lock:
            if self.anchor_map_xy is None or self.anchor_local_xy is None:
                return None
            anchor_map_x, anchor_map_y = self.anchor_map_xy
            anchor_local_x, anchor_local_y = self.anchor_local_xy
            matrix = None if self.map_to_local_matrix is None else self.map_to_local_matrix.copy()

        dx = map_xy[0] - anchor_map_x
        dy = map_xy[1] - anchor_map_y
        if matrix is None:
            if not self.args.skip_transform_calibration:
                return None
            dx *= self.args.map_x_sign
            dy *= self.args.map_y_sign
            yaw = math.radians(self.args.map_to_local_yaw_deg)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            local_dx = cos_yaw * dx - sin_yaw * dy
            local_dy = sin_yaw * dx + cos_yaw * dy
        else:
            local_dx, local_dy = matrix @ np.array([dx, dy], dtype=np.float64)
        return anchor_local_x + local_dx, anchor_local_y + local_dy

    def set_id15(
        self,
        record: aruco.MarkerRecord,
        map_xy: tuple[float, float],
        local_xy: tuple[float, float] | None,
    ) -> None:
        with self.lock:
            self.id15_record = dict(record)
            self.id15_map_xy = map_xy
            self.id15_local_xy = local_xy
            self.id15_seen_time = time.time()

    def set_error(self, exc: Exception) -> None:
        with self.lock:
            self.latest_error = str(exc)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "phase": self.phase,
                "target_point": self.target_point,
                "frame_id": self.frame_id,
                "last_result": self.last_result,
                "last_visible_ids": list(self.last_visible_ids),
                "anchor_map_xy": self.anchor_map_xy,
                "anchor_local_xy": self.anchor_local_xy,
                "map_to_local_matrix": None
                if self.map_to_local_matrix is None
                else self.map_to_local_matrix.tolist(),
                "id15_map_xy": self.id15_map_xy,
                "id15_local_xy": self.id15_local_xy,
                "id15_seen_time": self.id15_seen_time,
                "latest_error": self.latest_error,
            }

    def id15_ready(self) -> bool:
        with self.lock:
            return self.id15_local_xy is not None

    def id15_target(self, height: float) -> Point3 | None:
        with self.lock:
            if self.id15_local_xy is None:
                return None
            return self.id15_local_xy[0], self.id15_local_xy[1], height


def timestamped_default_paths() -> dict[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "events": "flights/{}_id15_events.csv".format(stamp),
        "localization": "flights/{}_id15_localization.csv".format(stamp),
        "aruco": "flights/{}_id15_aruco.csv".format(stamp),
        "preview": "flights/{}_id15_preview.jpg".format(stamp),
    }


def create_localizer(args: argparse.Namespace) -> OrbRansacLocalizer:
    return OrbRansacLocalizer(
        reference_path=args.reference,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        feature="orb",
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


def create_camera(args: argparse.Namespace, sdk2):
    if args.camera_source == "sdk2":
        camera = Sdk2Camera(sdk2, args.sdk2_camera_type, args.camera_timeout)
        camera_name = args.sdk2_camera_type.upper()
    else:
        camera = OpenCvCamera(args.camera_index)
        camera_name = "opencv:{}".format(args.camera_index)

    if args.undistort:
        calibration = load_camera_calibration(args.calibration, alpha=args.calibration_alpha)
        camera = UndistortedCamera(camera, calibration)
        camera_name += "+undistorted"
    return camera, camera_name


def local_position(drone) -> Point3:
    position = drone.get_local_position_lps()
    if position is None or len(position) < 3:
        raise RuntimeError("get_local_position_lps() returned no valid position")
    return float(position[0]), float(position[1]), float(position[2])


def project_marker_to_map(
    record: aruco.MarkerRecord,
    homography_frame_to_ref: np.ndarray | None,
    localizer: OrbRansacLocalizer,
) -> tuple[float, float] | None:
    if homography_frame_to_ref is None:
        return None

    center_frame = np.float32([[record["center_px"]]])
    center_ref = cv2.perspectiveTransform(center_frame, homography_frame_to_ref)[0, 0]
    if not np.all(np.isfinite(center_ref)):
        return None

    ref_x, ref_y = float(center_ref[0]), float(center_ref[1])
    if not (0.0 <= ref_x <= localizer.ref_w and 0.0 <= ref_y <= localizer.ref_h):
        return None
    return localizer.ref_pixel_to_map_m(ref_x, ref_y)


def draw_overlay(
    frame: np.ndarray,
    result: LocalizeResult,
    records: list[aruco.MarkerRecord],
    visible_ids: list[int],
    state: MissionState,
) -> np.ndarray:
    overlay = frame.copy()
    for record in records:
        aruco_hand_check.draw_marker(overlay, record)

    snapshot = state.snapshot()
    color = (0, 220, 0) if result.ok else (0, 0, 255)
    xy_text = "orb: no fix"
    if result.x_m is not None and result.y_m is not None:
        xy_text = "orb: x={:.2f} y={:.2f}".format(result.x_m, result.y_m)
    id15_text = "id15: not found"
    if snapshot["id15_map_xy"] is not None:
        map_x, map_y = snapshot["id15_map_xy"]
        id15_text = "id15 map: x={:.2f} y={:.2f}".format(map_x, map_y)
    if snapshot["id15_local_xy"] is not None:
        local_x, local_y = snapshot["id15_local_xy"]
        id15_text += " local: x={:.2f} y={:.2f}".format(local_x, local_y)

    lines = [
        "phase: {} frame: {}".format(snapshot["phase"], snapshot["frame_id"]),
        "{} status: {}".format(xy_text, result.message),
        "visible ids: {} detected_word: {}".format(visible_ids, aruco.current_detected_word() or "-"),
        id15_text,
    ]
    if snapshot["target_point"] is not None:
        tx, ty, tz = snapshot["target_point"]
        lines.append("target: x={:.2f} y={:.2f} z={:.2f}".format(tx, ty, tz))
    lines.append("map->local: {}".format("auto" if snapshot["map_to_local_matrix"] is not None else "not ready"))
    if snapshot["latest_error"]:
        lines.append("error: {}".format(snapshot["latest_error"]))

    panel_height = 28 + 24 * len(lines)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], panel_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(overlay, line, (12, 26 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return overlay


def vision_loop(
    camera,
    localizer: OrbRansacLocalizer,
    state: MissionState,
    args: argparse.Namespace,
    stop_event: threading.Event,
    preview_store: aruco_hand_check.PreviewFrameStore,
) -> None:
    dictionary = aruco.resolve_aruco_dictionary(args.aruco_dict)
    params = aruco.create_detector_parameters()
    detector = aruco.create_detector(dictionary, params)
    camera_matrix, dist_coeffs = aruco.load_calibration(args.calibration)
    seen_ids: set[int] = set()
    last_preview_jpeg_time = 0.0

    try:
        while not stop_event.is_set():
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            result, homography, processed_frame = localizer.estimate(frame)
            corners, ids, _ = aruco.detect_markers(processed_frame, dictionary, params, detector)
            records: list[aruco.MarkerRecord] = []
            visible_ids: list[int] = []

            if ids is not None:
                for index, raw_id in enumerate(ids.flatten()):
                    marker_id = int(raw_id)
                    visible_ids.append(marker_id)
                    first_seen = marker_id not in seen_ids
                    seen_ids.add(marker_id)
                    frame_id = int(state.snapshot()["frame_id"])
                    record = aruco.build_marker_record(
                        frame_id,
                        marker_id,
                        corners[index],
                        camera_matrix,
                        dist_coeffs,
                        first_seen,
                    )
                    record["center_map_m"] = None
                    record["target_local_xy"] = None
                    marker_map_xy = project_marker_to_map(record, homography, localizer) if result.ok else None
                    if marker_map_xy is not None:
                        marker_local_xy = state.map_to_local_xy(marker_map_xy)
                        record["center_map_m"] = [marker_map_xy[0], marker_map_xy[1]]
                        record["target_local_xy"] = (
                            None if marker_local_xy is None else [marker_local_xy[0], marker_local_xy[1]]
                        )
                        if marker_id == aruco.DELIVERY_MARKER_ID and marker_local_xy is not None:
                            state.set_id15(record, marker_map_xy, marker_local_xy)
                    aruco.emit_marker(record)
                    records.append(record)

            state.set_orb_result(result, visible_ids)
            row = {
                "timestamp": time.time(),
                "frame_id": state.snapshot()["frame_id"],
                "phase": state.snapshot()["phase"],
                "ok": result.ok,
                "message": result.message,
                "x_m": result.x_m,
                "y_m": result.y_m,
                "raw_x_m": result.raw_x_m,
                "raw_y_m": result.raw_y_m,
                "good_matches": result.good_matches,
                "inliers": result.inliers,
                "inlier_ratio": result.inlier_ratio,
                "homography_area": result.homography_area,
                "visible_ids": json.dumps(visible_ids),
                "id15_map_xy": json.dumps(state.snapshot()["id15_map_xy"]),
                "id15_local_xy": json.dumps(state.snapshot()["id15_local_xy"]),
            }
            append_csv(args.localization_csv, row)

            overlay = draw_overlay(processed_frame, result, records, visible_ids, state)
            preview_store.update(overlay, args.preview_quality)
            last_preview_jpeg_time = aruco_hand_check.save_preview_jpeg(
                args.preview_jpeg,
                overlay,
                last_preview_jpeg_time,
                args.preview_jpeg_interval,
            )
            time.sleep(0.02)
    except Exception as exc:
        state.set_error(exc)
        stop_event.set()


def wait_for_orb_fix(state: MissionState, timeout: float, stop_event: threading.Event) -> tuple[float, float]:
    start = time.monotonic()
    while not stop_event.is_set() and time.monotonic() - start < timeout:
        snapshot = state.snapshot()
        result = snapshot["last_result"]
        if isinstance(result, LocalizeResult) and result.ok and result.x_m is not None and result.y_m is not None:
            return result.x_m, result.y_m
        time.sleep(0.05)
    raise RuntimeError("ORB localization was not acquired within {:.1f}s".format(timeout))


def wait_for_stable_orb_fix(
    state: MissionState,
    timeout: float,
    stop_event: threading.Event,
    samples: int,
    max_spread: float,
) -> tuple[float, float]:
    start = time.monotonic()
    observed: list[tuple[float, float]] = []
    last_frame_id = -1
    while not stop_event.is_set() and time.monotonic() - start < timeout:
        snapshot = state.snapshot()
        result = snapshot["last_result"]
        frame_id = int(snapshot["frame_id"])
        if (
            frame_id != last_frame_id
            and isinstance(result, LocalizeResult)
            and result.ok
            and result.x_m is not None
            and result.y_m is not None
        ):
            observed.append((float(result.x_m), float(result.y_m)))
            observed = observed[-max(samples, 1) :]
            last_frame_id = frame_id
            if len(observed) >= samples:
                arr = np.asarray(observed, dtype=np.float64)
                median = np.median(arr, axis=0)
                spread = float(np.max(np.linalg.norm(arr - median, axis=1)))
                if spread <= max_spread:
                    return float(median[0]), float(median[1])
        time.sleep(0.05)
    raise RuntimeError("stable ORB localization was not acquired within {:.1f}s".format(timeout))


def wait_for_point_or_id15(
    drone,
    state: MissionState,
    timeout: float,
    poll_interval: float,
    stop_event: threading.Event,
) -> bool:
    start = time.monotonic()
    reached = False
    while not stop_event.is_set() and time.monotonic() - start < timeout:
        if state.id15_ready():
            return True
        if hasattr(drone, "point_reached"):
            try:
                if drone.point_reached():
                    reached = True
                    break
            except Exception:
                break
        time.sleep(poll_interval)
    return reached


def command_and_wait(
    drone,
    point: Point3,
    previous_point: Point3,
    args: argparse.Namespace,
    stop_event: threading.Event,
    state: MissionState,
    event_logger: FlightEventLogger,
    phase: str,
    break_on_id15: bool = False,
) -> bool:
    state.set_phase(phase, point)
    point_time = estimate_move_time(previous_point, point, args.speed)
    event_logger.log(
        "{}_start".format(phase),
        phase,
        "Sending local point",
        target_point=point,
        details={"point_time": point_time},
    )
    command_local_point(drone, point[0], point[1], point[2], yaw=args.yaw, point_time=point_time)

    start = time.monotonic()
    reached = False
    while not stop_event.is_set() and time.monotonic() - start < args.move_timeout:
        if break_on_id15 and state.id15_ready():
            event_logger.log("{}_interrupted_id15".format(phase), phase, "ID15 detected during movement", target_point=point)
            return True
        if hasattr(drone, "point_reached"):
            try:
                if drone.point_reached():
                    reached = True
                    break
            except Exception as exc:
                event_logger.log("{}_point_reached_error".format(phase), phase, str(exc), target_point=point)
                break
        time.sleep(args.poll_interval)

    event_logger.log(
        "{}_complete".format(phase) if reached else "{}_timeout".format(phase),
        phase,
        "Local point reached" if reached else "Local point was not confirmed before timeout",
        target_point=point,
        details={"reached": reached},
    )
    return reached


def calibrate_map_to_local_transform(
    drone,
    start_local: Point3,
    args: argparse.Namespace,
    stop_event: threading.Event,
    state: MissionState,
    event_logger: FlightEventLogger,
) -> Point3:
    state.set_phase("map_transform_anchor", start_local)
    anchor_map = wait_for_stable_orb_fix(
        state,
        args.orb_timeout,
        stop_event,
        args.transform_samples,
        args.transform_max_spread,
    )
    anchor_local = local_position(drone)
    state.set_anchor(anchor_map, (anchor_local[0], anchor_local[1]))
    event_logger.log(
        "map_transform_anchor",
        "map_transform",
        "Map/local anchor locked",
        target_point=anchor_local,
        details={"anchor_map_xy": anchor_map, "anchor_local_xy": (anchor_local[0], anchor_local[1])},
    )

    if args.skip_transform_calibration:
        event_logger.log(
            "map_transform_manual",
            "map_transform",
            "Skipping automatic map/local transform calibration",
            target_point=anchor_local,
            details={
                "map_to_local_yaw_deg": args.map_to_local_yaw_deg,
                "map_x_sign": args.map_x_sign,
                "map_y_sign": args.map_y_sign,
            },
        )
        return anchor_local

    previous = anchor_local
    probes: list[tuple[str, Point3]] = [
        (
            "map_transform_x",
            (anchor_local[0] + args.transform_calibration_distance, anchor_local[1], args.height),
        ),
        (
            "map_transform_y",
            (anchor_local[0], anchor_local[1] + args.transform_calibration_distance, args.height),
        ),
    ]
    map_points: list[tuple[float, float]] = []
    local_points: list[tuple[float, float]] = []

    for phase, point in probes:
        command_and_wait(drone, point, previous, args, stop_event, state, event_logger, phase)
        time.sleep(args.transform_settle)
        map_xy = wait_for_stable_orb_fix(
            state,
            args.orb_timeout,
            stop_event,
            args.transform_samples,
            args.transform_max_spread,
        )
        local_xy = local_position(drone)
        map_points.append(map_xy)
        local_points.append((local_xy[0], local_xy[1]))
        event_logger.log(
            "{}_measured".format(phase),
            "map_transform",
            "Map/local probe measured",
            target_point=local_xy,
            details={"map_xy": map_xy, "local_xy": (local_xy[0], local_xy[1])},
        )
        previous = local_xy

    map_delta_x = np.array(map_points[0], dtype=np.float64) - np.array(anchor_map, dtype=np.float64)
    map_delta_y = np.array(map_points[1], dtype=np.float64) - np.array(anchor_map, dtype=np.float64)
    local_delta_x = np.array(local_points[0], dtype=np.float64) - np.array(anchor_local[:2], dtype=np.float64)
    local_delta_y = np.array(local_points[1], dtype=np.float64) - np.array(anchor_local[:2], dtype=np.float64)
    map_matrix = np.column_stack((map_delta_x, map_delta_y))
    local_matrix = np.column_stack((local_delta_x, local_delta_y))

    determinant = float(np.linalg.det(map_matrix))
    condition = float(np.linalg.cond(map_matrix))
    if abs(determinant) < args.transform_min_determinant or not np.isfinite(condition) or condition > args.transform_max_condition:
        raise RuntimeError(
            "cannot estimate map/local transform: determinant={:.6f}, condition={:.2f}".format(
                determinant,
                condition,
            )
        )

    transform = local_matrix @ np.linalg.inv(map_matrix)
    state.set_map_to_local_matrix(transform)
    event_logger.log(
        "map_transform_calibrated",
        "map_transform",
        "Automatic map/local transform calibrated",
        target_point=previous,
        details={
            "matrix": transform.tolist(),
            "determinant": determinant,
            "condition": condition,
            "anchor_map_xy": anchor_map,
            "anchor_local_xy": (anchor_local[0], anchor_local[1]),
        },
    )
    return_point = (anchor_local[0], anchor_local[1], args.height)
    command_and_wait(
        drone,
        return_point,
        previous,
        args,
        stop_event,
        state,
        event_logger,
        "map_transform_return",
    )
    return local_position(drone)


def build_scan_points(center: Point3, args: argparse.Namespace) -> list[Point3]:
    cx, cy, cz = center
    half = args.scan_area_size / 2.0
    if args.scan_grid <= 1:
        return [(cx, cy, cz)]

    xs = [cx - half + args.scan_area_size * index / (args.scan_grid - 1) for index in range(args.scan_grid)]
    ys = [cy - half + args.scan_area_size * index / (args.scan_grid - 1) for index in range(args.scan_grid)]
    points: list[Point3] = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((x, y, args.height))
    return points


def land(drone, event_logger: FlightEventLogger, phase: str) -> None:
    event_logger.log("landing_start", phase, "Sending land command")
    drone.land()
    event_logger.log("landing_sent", phase, "Land command sent")


def run(args: argparse.Namespace) -> int:
    Path("flights").mkdir(exist_ok=True)
    aruco.CSV_PATH = args.aruco_csv
    aruco.ARUCO_DICTIONARY = args.aruco_dict
    aruco.CALIBRATION_FILE = args.calibration
    aruco.DELIVERY_MARKER_ID = args.delivery_id
    aruco.set_forbidden_marker(args.forbidden_id)
    aruco.LATEST_MARKERS.clear()

    event_logger = FlightEventLogger(args.events_log)
    stop_event = threading.Event()
    command_state = FlightCommandState()
    if not args.no_command_listener:
        start_command_listener(stop_event, command_state=command_state, event_logger=event_logger)

    preview_store = aruco_hand_check.PreviewFrameStore()
    preview_server = None
    sdk2 = import_pioneer_sdk2()
    drone = create_pioneer(sdk2)
    camera, camera_name = create_camera(args, sdk2)
    localizer = create_localizer(args)
    state = MissionState(args)
    vision_thread = threading.Thread(
        target=vision_loop,
        args=(camera, localizer, state, args, stop_event, preview_store),
        daemon=True,
    )

    flight_started = False
    is_armed = False

    try:
        if args.preview_port > 0:
            preview_server = aruco_hand_check.start_preview_server(args.preview_host, args.preview_port, preview_store)
            address = "http://{}:{}/".format(args.preview_host, args.preview_port)
            if args.preview_host in ("0.0.0.0", "::"):
                address = "http://<drone-ip>:{}/".format(args.preview_port)
            print("Live ID15 mission preview: {}".format(address))

        event_logger.log(
            "mission_start",
            "setup",
            "ID15 mission started",
            details={"camera": camera_name, "reference": args.reference, "height": args.height},
        )
        vision_thread.start()
        check_battery_or_abort(drone, args.min_battery_voltage, args.battery_check_retries, args.battery_check_delay)

        state.set_phase("arm")
        event_logger.log("arm_start", "arm", "Sending arm command")
        if hasattr(drone, "arm"):
            armed = drone.arm(timeout=5, retries=1)
            if armed is False:
                raise RuntimeError("pioneer.arm() returned False")
            is_armed = True
        event_logger.log("arm_complete", "arm", "Drone armed")

        state.set_phase("takeoff")
        event_logger.log("takeoff_start", "takeoff", "Sending takeoff command")
        takeoff = drone.takeoff()
        if takeoff is False:
            raise RuntimeError("pioneer.takeoff() returned False")
        flight_started = True
        event_logger.log("takeoff_sent", "takeoff", "Takeoff command sent")
        time.sleep(args.takeoff_wait)

        start_local = local_position(drone)
        climb_point = (start_local[0], start_local[1], args.height)
        command_and_wait(drone, climb_point, start_local, args, stop_event, state, event_logger, "climb")
        start_local = local_position(drone)

        previous = calibrate_map_to_local_transform(drone, start_local, args, stop_event, state, event_logger)
        previous = (previous[0], previous[1], args.height)
        for index, point in enumerate(build_scan_points(previous, args), 1):
            if stop_event.is_set() or state.id15_ready():
                break
            event_logger.log("scan_point", "scan", "Going to scan point {}".format(index), point_index=index, target_point=point)
            command_and_wait(
                drone,
                point,
                previous,
                args,
                stop_event,
                state,
                event_logger,
                "scan",
                break_on_id15=True,
            )
            previous = point
            if not state.id15_ready() and args.scan_settle > 0:
                state.set_phase("scan_settle", point)
                time.sleep(args.scan_settle)

        if not state.id15_ready():
            raise RuntimeError("ID15 was not found during scan")

        id15_above = state.id15_target(args.height)
        if id15_above is None:
            raise RuntimeError("ID15 target local coordinates are not available")

        command_and_wait(drone, id15_above, previous, args, stop_event, state, event_logger, "go_to_id15")
        time.sleep(args.target_settle)

        low_point = (id15_above[0], id15_above[1], args.landing_approach_height)
        command_and_wait(drone, low_point, id15_above, args, stop_event, state, event_logger, "descend_to_id15")
        land(drone, event_logger, "land_on_id15")
        flight_started = False
        time.sleep(args.landed_wait)

        if stop_event.is_set():
            return 130

        state.set_phase("retakeoff")
        if hasattr(drone, "arm"):
            event_logger.log("retakeoff_arm_start", "retakeoff", "Arming before retakeoff")
            armed = drone.arm(timeout=5, retries=1)
            if armed is False:
                raise RuntimeError("pioneer.arm() returned False before retakeoff")
            is_armed = True
            event_logger.log("retakeoff_arm_complete", "retakeoff", "Drone armed before retakeoff")
        event_logger.log("retakeoff_start", "retakeoff", "Taking off again after ID15 landing")
        takeoff = drone.takeoff()
        if takeoff is False:
            raise RuntimeError("pioneer.takeoff() returned False on retakeoff")
        flight_started = True
        time.sleep(args.takeoff_wait)

        current = local_position(drone)
        return_high = (start_local[0], start_local[1], args.height)
        command_and_wait(drone, return_high, current, args, stop_event, state, event_logger, "return_home")
        land(drone, event_logger, "final_land")
        flight_started = False
        time.sleep(args.landed_wait)

        if hasattr(drone, "disarm"):
            state.set_phase("disarm")
            event_logger.log("disarm_start", "disarm", "Sending disarm command")
            drone.disarm()
            is_armed = False
            event_logger.log("disarm_complete", "disarm", "Drone disarmed")
        event_logger.log("mission_complete", "complete", "ID15 mission completed")
        return 0

    except KeyboardInterrupt:
        event_logger.log("keyboard_interrupt", "interrupt", "Keyboard interrupt received")
        return 130
    except Exception as exc:
        state.set_error(exc)
        event_logger.log("mission_error", "error", "Mission error: {}".format(exc), details={"error": str(exc)})
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        if flight_started and hasattr(drone, "land"):
            try:
                land(drone, event_logger, "cleanup")
            except Exception as exc:
                event_logger.log("cleanup_land_failed", "cleanup", str(exc), details={"error": str(exc)})
        elif is_armed and hasattr(drone, "disarm"):
            try:
                drone.disarm()
            except Exception as exc:
                event_logger.log("cleanup_disarm_failed", "cleanup", str(exc), details={"error": str(exc)})

        vision_thread.join(timeout=2.0)
        camera.close()
        if preview_server is not None:
            preview_server[0].shutdown()
            preview_server[0].server_close()
        preview_store.stop()
        event_logger.log("cleanup_complete", "cleanup", "Cleanup completed")


def build_parser() -> argparse.ArgumentParser:
    defaults = timestamped_default_paths()
    parser = argparse.ArgumentParser(description="Take off, find ArUco ID15 using ORB map coordinates, land on it, and return.")
    parser.add_argument("--reference", default="map.jpg")
    parser.add_argument("--map-width-m", type=float, default=3.0)
    parser.add_argument("--map-height-m", type=float, default=3.0)
    parser.add_argument("--camera-source", choices=["sdk2", "opencv"], default="sdk2")
    parser.add_argument("--sdk2-camera-type", default="OPT")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--calibration", default="data.yml")
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--calibration-alpha", type=float, default=0.0)
    parser.add_argument("--aruco-dict", default=aruco.ARUCO_DICTIONARY)
    parser.add_argument("--delivery-id", type=int, default=15)
    parser.add_argument("--forbidden-id", type=int)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--landing-approach-height", type=float, default=0.35)
    parser.add_argument("--speed", type=float, default=0.18)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--takeoff-wait", type=float, default=3.0)
    parser.add_argument("--landed-wait", type=float, default=3.0)
    parser.add_argument("--target-settle", type=float, default=1.0)
    parser.add_argument("--move-timeout", type=float, default=25.0)
    parser.add_argument("--poll-interval", type=float, default=0.15)
    parser.add_argument("--scan-area-size", type=float, default=1.2)
    parser.add_argument("--scan-grid", type=int, default=3)
    parser.add_argument("--scan-settle", type=float, default=0.3)
    parser.add_argument("--orb-timeout", type=float, default=15.0)
    parser.add_argument("--skip-transform-calibration", action="store_true")
    parser.add_argument("--transform-calibration-distance", type=float, default=0.3)
    parser.add_argument("--transform-samples", type=int, default=6)
    parser.add_argument("--transform-max-spread", type=float, default=0.12)
    parser.add_argument("--transform-settle", type=float, default=0.6)
    parser.add_argument("--transform-min-determinant", type=float, default=0.01)
    parser.add_argument("--transform-max-condition", type=float, default=8.0)
    parser.add_argument("--map-to-local-yaw-deg", type=float, default=0.0)
    parser.add_argument("--map-x-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--map-y-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--nfeatures", type=int, default=1500)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=18)
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--reference-max-size", type=int, default=1600)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-tile", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.55)
    parser.add_argument("--min-homography-area-m2", type=float, default=0.03)
    parser.add_argument("--max-homography-area-m2", type=float, default=2.2)
    parser.add_argument("--max-position-jump", type=float, default=0.8)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=8002)
    parser.add_argument("--preview-quality", type=int, default=80)
    parser.add_argument("--preview-jpeg", default=defaults["preview"])
    parser.add_argument("--preview-jpeg-interval", type=float, default=0.2)
    parser.add_argument("--events-log", default=defaults["events"])
    parser.add_argument("--localization-csv", default=defaults["localization"])
    parser.add_argument("--aruco-csv", default=defaults["aruco"])
    parser.add_argument("--min-battery-voltage", type=float, default=0.0)
    parser.add_argument("--battery-check-retries", type=int, default=3)
    parser.add_argument("--battery-check-delay", type=float, default=0.5)
    parser.add_argument("--no-command-listener", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0:
        raise ValueError("--height must be positive")
    if args.landing_approach_height <= 0 or args.landing_approach_height >= args.height:
        raise ValueError("--landing-approach-height must be positive and lower than --height")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.scan_area_size <= 0:
        raise ValueError("--scan-area-size must be positive")
    if args.scan_grid <= 0:
        raise ValueError("--scan-grid must be positive")
    if args.transform_calibration_distance <= 0:
        raise ValueError("--transform-calibration-distance must be positive")
    if args.transform_samples <= 0:
        raise ValueError("--transform-samples must be positive")
    if args.transform_max_spread <= 0:
        raise ValueError("--transform-max-spread must be positive")
    if args.transform_settle < 0:
        raise ValueError("--transform-settle must be non-negative")
    if args.transform_min_determinant <= 0:
        raise ValueError("--transform-min-determinant must be positive")
    if args.transform_max_condition <= 1:
        raise ValueError("--transform-max-condition must be greater than 1")
    if args.preview_port < 0 or args.preview_port > 65535:
        raise ValueError("--preview-port must be between 0 and 65535")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
