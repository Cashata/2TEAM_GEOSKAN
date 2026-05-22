#!/usr/bin/env python3
"""Continuous recording, CSV logging, and mission video overlays."""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from geoscan_mission.vision.aruco import ArucoDetector, ArucoMarker
from geoscan_mission.vision.localization import LocalizeResult, OrbRansacLocalizer


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
    phase: str,
    aruco_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    row = asdict(result)
    row["timestamp"] = time.time()
    row["point_index"] = point_index
    row["frame_id"] = frame_id
    row["phase"] = phase
    row["camera_type"] = camera_type
    row["height"] = target_point[2]
    row["yaw"] = yaw
    row["target_x_m"] = target_point[0]
    row["target_y_m"] = target_point[1]
    row["target_z_m"] = target_point[2]
    row["target_yaw"] = yaw
    if aruco_summary is not None:
        row.update(aruco_summary)
    return row


def video_path_enabled(path: str | None) -> bool:
    return bool(path and path.strip() and path.strip().lower() != "none")


def map_point_to_pixel(localizer: OrbRansacLocalizer, x_m: float, y_m: float) -> tuple[int, int]:
    px = int(round(x_m / localizer.map_width_m * localizer.ref_w))
    py = int(round(y_m / localizer.map_height_m * localizer.ref_h))
    return px, py


def project_aruco_markers(
    markers: list[ArucoMarker],
    homography_frame_to_ref: np.ndarray | None,
    localizer: OrbRansacLocalizer,
    detector: ArucoDetector,
) -> None:
    if homography_frame_to_ref is None:
        return

    for marker in markers:
        center_frame = np.float32([[marker.center_px]])
        center_ref = cv2.perspectiveTransform(center_frame, homography_frame_to_ref)[0, 0]
        if not np.all(np.isfinite(center_ref)):
            continue

        ref_x, ref_y = float(center_ref[0]), float(center_ref[1])
        marker.center_ref_px = [ref_x, ref_y]
        if 0.0 <= ref_x <= localizer.ref_w and 0.0 <= ref_y <= localizer.ref_h:
            map_x, map_y = localizer.ref_pixel_to_map_m(ref_x, ref_y)
            marker.center_map_m = [map_x, map_y]
        detector.remember_projection(marker)


def aruco_frame_summary(markers: list[ArucoMarker], detector: ArucoDetector) -> dict[str, object]:
    return {
        "aruco_seen_ids": [marker.marker_id for marker in markers],
        "aruco_new_ids": [marker.marker_id for marker in markers if marker.first_seen],
        "aruco_word": detector.get_word(),
        "aruco_allowed_ids": detector.found_ids("allowed"),
        "aruco_forbidden_ids": detector.found_ids("forbidden"),
        "aruco_markers_json": json.dumps([marker.as_dict() for marker in markers], ensure_ascii=False),
    }


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

    def camera_overlay(
        self,
        frame_bgr: np.ndarray,
        result: LocalizeResult,
        point_index: int,
        frame_id: int,
        aruco_markers: list[ArucoMarker] | None = None,
        aruco_summary: dict[str, object] | None = None,
    ) -> np.ndarray:
        overlay = frame_bgr.copy()
        h, w = overlay.shape[:2]
        color = (0, 200, 0) if result.ok else (0, 0, 255)
        status = "OK" if result.ok else "FAIL"
        cv2.circle(overlay, (w // 2, h // 2), 7, color, -1)

        if aruco_markers:
            for marker in aruco_markers:
                points = np.array(marker.corners_px, dtype=np.int32)
                cv2.polylines(overlay, [points], True, (255, 180, 0), 2)
                center_x, center_y = marker.center_px
                cv2.circle(overlay, (int(round(center_x)), int(round(center_y))), 5, (255, 180, 0), -1)
                label = "ID {} {}".format(marker.marker_id, marker.target_type)
                cv2.putText(
                    overlay,
                    label,
                    (int(round(center_x)) + 8, int(round(center_y)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 180, 0),
                    2,
                )

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
        if aruco_summary is not None:
            lines.append(
                "aruco ids={} word={}".format(
                    aruco_summary["aruco_seen_ids"],
                    aruco_summary["aruco_word"] or "-",
                )
            )
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

    def write(
        self,
        frame_bgr: np.ndarray,
        result: LocalizeResult,
        point_index: int,
        frame_id: int,
        aruco_markers: list[ArucoMarker] | None = None,
        aruco_summary: dict[str, object] | None = None,
    ) -> None:
        if self.camera_path:
            camera_frame = self.camera_overlay(
                frame_bgr,
                result,
                point_index,
                frame_id,
                aruco_markers=aruco_markers,
                aruco_summary=aruco_summary,
            )
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


class ContinuousFlightRecorder:
    def __init__(
        self,
        camera,
        localizer: OrbRansacLocalizer,
        video_logger: FlightVideoLogger,
        csv_path: str | None,
        debug_dir: str | None,
        camera_type: str,
        yaw: float,
        stop_event: threading.Event,
        aruco_detector: ArucoDetector | None = None,
    ) -> None:
        self.camera = camera
        self.localizer = localizer
        self.video_logger = video_logger
        self.csv_path = csv_path
        self.debug_dir = debug_dir
        self.camera_type = camera_type
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.frame_id = 0
        self.error: Exception | None = None
        self.point_index = 0
        self.target_point = (0.0, 0.0, 0.0)
        self.yaw = yaw
        self.phase = "preflight"
        self.reset_tracking = True
        self.aruco_detector = aruco_detector

    def set_context(
        self,
        point_index: int,
        target_point: tuple[float, float, float],
        yaw: float,
        phase: str,
        reset_tracking: bool = False,
    ) -> None:
        with self.lock:
            self.point_index = point_index
            self.target_point = target_point
            self.yaw = yaw
            self.phase = phase
            if reset_tracking:
                self.reset_tracking = True

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join()

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("Continuous flight recorder failed: {}".format(self.error)) from self.error

    def snapshot_context(self) -> tuple[int, tuple[float, float, float], float, str, bool]:
        with self.lock:
            context = (self.point_index, self.target_point, self.yaw, self.phase, self.reset_tracking)
            self.reset_tracking = False
            return context

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                point_index, target_point, yaw, phase, reset_tracking = self.snapshot_context()
                if reset_tracking:
                    self.localizer.start_waypoint()

                frame = self.camera.read()
                if frame is None:
                    if getattr(self.camera, "finished", False):
                        print("video input ended")
                        self.stop_event.set()
                        break
                    print("camera frame is empty", file=sys.stderr)
                    time.sleep(0.05)
                    continue

                result, homography, processed_frame = self.localizer.estimate(frame)
                frame_id = self.frame_id
                self.frame_id += 1

                aruco_markers: list[ArucoMarker] = []
                aruco_summary = None
                if self.aruco_detector is not None:
                    aruco_markers = self.aruco_detector.process_frame(processed_frame)
                    if result.ok:
                        project_aruco_markers(
                            aruco_markers,
                            homography,
                            self.localizer,
                            self.aruco_detector,
                        )
                    aruco_summary = aruco_frame_summary(aruco_markers, self.aruco_detector)

                row = result_row(
                    point_index,
                    frame_id,
                    result,
                    self.camera_type,
                    target_point,
                    yaw,
                    phase,
                    aruco_summary=aruco_summary,
                )
                print(json.dumps(row, ensure_ascii=False))
                append_csv(self.csv_path, row)
                self.video_logger.write(
                    processed_frame,
                    result,
                    point_index,
                    frame_id,
                    aruco_markers=aruco_markers,
                    aruco_summary=aruco_summary,
                )

                if self.debug_dir:
                    out = Path(self.debug_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    debug = self.localizer.draw_debug(processed_frame, result, homography)
                    filename = "debug_p{:03d}_f{:06d}.jpg".format(point_index, frame_id)
                    cv2.imwrite(str(out / filename), debug)

                time.sleep(0.03)

        except Exception as exc:
            self.error = exc
            print("ERROR: continuous flight recorder failed: {}".format(exc), file=sys.stderr)
            self.stop_event.set()
