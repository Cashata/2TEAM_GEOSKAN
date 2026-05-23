#!/usr/bin/env python3
"""Live ArUco hand-check preview.

This script does not fly the drone. It opens the selected camera, detects the
mission ArUco markers from aruco_tracker_standalone.py, writes records through
that same standalone tracker, and serves an MJPEG preview like calibration.py.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import aruco_tracker_standalone as aruco


class PreviewFrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg_bytes: bytes | None = None
        self.stopped = False

    def update(self, frame: np.ndarray, quality: int) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self.condition:
            self.jpeg_bytes = encoded.tobytes()
            self.condition.notify_all()

    def wait_jpeg(self, timeout: float) -> bytes | None:
        with self.condition:
            if self.jpeg_bytes is None and not self.stopped:
                self.condition.wait(timeout=timeout)
            return self.jpeg_bytes

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


def start_preview_server(host: str, port: int, store: PreviewFrameStore):
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                html = (
                    "<!doctype html><html><head><title>ArUco hand check</title>"
                    "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                    "img{max-width:100vw;max-height:100vh;display:block;margin:auto}</style>"
                    "</head><body><img src='/stream.mjpg'></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/latest.jpg":
                jpeg = store.wait_jpeg(timeout=1.0)
                if jpeg is None:
                    self.send_error(503, "preview frame is not ready")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while not store.stopped:
                    jpeg = store.wait_jpeg(timeout=1.0)
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return

            self.send_error(404)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), PreviewHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def save_preview_jpeg(path: str | None, frame: np.ndarray, last_write: float, interval: float) -> float:
    if not path:
        return last_write
    now = time.monotonic()
    if now - last_write < interval:
        return last_write
    cv2.imwrite(path, frame)
    return now


def open_camera(args: argparse.Namespace):
    aruco.CAMERA_SOURCE = args.camera_source
    aruco.SDK2_CAMERA_TYPE = args.sdk2_camera_type
    aruco.CAMERA_INDEX = args.camera_index
    aruco.CAMERA_TIMEOUT = args.camera_timeout
    if args.camera_source == "sdk2":
        return aruco.open_sdk2_camera(), args.sdk2_camera_type.upper()
    return aruco.open_opencv_camera(), "opencv:{}".format(args.camera_index)


def configure_tracker(args: argparse.Namespace) -> None:
    aruco.ARUCO_DICTIONARY = args.aruco_dict
    aruco.CALIBRATION_FILE = args.calibration
    aruco.CSV_PATH = args.csv
    aruco.FRAME_LIMIT = args.frame_limit
    aruco.set_forbidden_marker(args.forbidden_id)
    aruco.LATEST_MARKERS.clear()


def draw_status(frame: np.ndarray, state: aruco.MarkerRecord, visible_ids: list[int]) -> None:
    lines = [
        "assignment: {}".format(state["assignment_word"]),
        "detected: {} ids={}".format(state["detected_word"] or "-", visible_ids),
        "allowed: {} expected={}".format(state["allowed_word"] or "-", state["expected_allowed_word"] or "-"),
        "all_found={} word_complete={}".format(state["all_markers_found"], state["word_complete"]),
        "delivery id15: found={} allowed={}".format(state["delivery_marker_found"], state["delivery_marker_allowed"]),
    ]
    if state["forbidden_marker_id"] is not None:
        lines.append("forbidden id: {}".format(state["forbidden_marker_id"]))

    width = frame.shape[1]
    panel_height = 28 + 24 * len(lines)
    cv2.rectangle(frame, (0, 0), (width, panel_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        color = (0, 220, 0)
        if "delivery id15" in line and not state["delivery_marker_found"]:
            color = (0, 180, 255)
        cv2.putText(frame, line, (12, 26 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_marker(frame: np.ndarray, record: aruco.MarkerRecord) -> None:
    points = np.array(record["corners_px"], dtype=np.int32)
    marker_id = int(record["id"])
    marker_type = str(record["type"])
    is_delivery = bool(record.get("delivery_candidate"))
    color = (0, 255, 0)
    if marker_type == "forbidden":
        color = (0, 0, 255)
    elif is_delivery:
        color = (255, 180, 0)

    cv2.polylines(frame, [points], True, color, 3)
    center_x, center_y = record["center_px"]
    cv2.circle(frame, (int(round(center_x)), int(round(center_y))), 5, color, -1)

    label = "ID {} {} {}".format(marker_id, record["letter"] or "?", marker_type)
    if is_delivery:
        label += " delivery"
    if record["marker_size_m"] is not None:
        label += " {:.0f}cm".format(float(record["marker_size_m"]) * 100.0)
    cv2.putText(
        frame,
        label,
        (int(round(center_x)) + 8, int(round(center_y)) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    tvec = record.get("tvec_m")
    if isinstance(tvec, list) and len(tvec) >= 3:
        pose_text = "x={:.2f} y={:.2f} z={:.2f}m".format(float(tvec[0]), float(tvec[1]), float(tvec[2]))
        cv2.putText(
            frame,
            pose_text,
            (int(round(center_x)) + 8, int(round(center_y)) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def process_frame(frame, frame_id: int, dictionary, params, detector, camera_matrix, dist_coeffs, seen_ids: set[int]):
    corners, ids, _ = aruco.detect_markers(frame, dictionary, params, detector)
    records: list[aruco.MarkerRecord] = []
    visible_ids: list[int] = []
    if ids is None:
        return records, visible_ids

    for index, raw_id in enumerate(ids.flatten()):
        marker_id = int(raw_id)
        visible_ids.append(marker_id)
        first_seen = marker_id not in seen_ids
        seen_ids.add(marker_id)
        record = aruco.build_marker_record(frame_id, marker_id, corners[index], camera_matrix, dist_coeffs, first_seen)
        aruco.emit_marker(record)
        records.append(record)
    return records, visible_ids


def run(args: argparse.Namespace) -> int:
    configure_tracker(args)
    camera, camera_name = open_camera(args)
    dictionary = aruco.resolve_aruco_dictionary(args.aruco_dict)
    params = aruco.create_detector_parameters()
    detector = aruco.create_detector(dictionary, params)
    camera_matrix, dist_coeffs = aruco.load_calibration(args.calibration)
    preview_store = PreviewFrameStore()
    preview_server = None
    seen_ids: set[int] = set()
    frame_id = 0
    last_preview_jpeg_time = 0.0

    try:
        if args.preview_port > 0:
            preview_server = start_preview_server(args.preview_host, args.preview_port, preview_store)
            address = "http://{}:{}/".format(args.preview_host, args.preview_port)
            if args.preview_host in ("0.0.0.0", "::"):
                address = "http://<drone-ip>:{}/".format(args.preview_port)
            print("Live ArUco preview: {}".format(address))

        print("Camera source: {}".format(camera_name))
        print("Assignment word: {}".format("".join(aruco.target_letter(marker_id) for marker_id in aruco.WORD_ORDER)))
        print("Delivery marker: ID {}".format(aruco.DELIVERY_MARKER_ID))

        while args.frame_limit <= 0 or frame_id < args.frame_limit:
            frame = aruco.read_frame(camera)
            if frame is None:
                time.sleep(0.05)
                continue

            records, visible_ids = process_frame(
                frame,
                frame_id,
                dictionary,
                params,
                detector,
                camera_matrix,
                dist_coeffs,
                seen_ids,
            )
            annotated = frame.copy()
            for record in records:
                draw_marker(annotated, record)
            draw_status(annotated, aruco.build_task_state(), visible_ids)
            preview_store.update(annotated, args.preview_quality)
            last_preview_jpeg_time = save_preview_jpeg(
                args.preview_jpeg,
                annotated,
                last_preview_jpeg_time,
                args.preview_jpeg_interval,
            )
            frame_id += 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 130
    finally:
        preview_store.stop()
        if preview_server is not None:
            preview_server[0].shutdown()
            preview_server[0].server_close()
        aruco.close_camera(camera)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live hand-check for mission ArUco markers.")
    parser.add_argument("--camera-source", choices=["sdk2", "opencv"], default="sdk2")
    parser.add_argument("--sdk2-camera-type", default="OPT")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--aruco-dict", default=aruco.ARUCO_DICTIONARY)
    parser.add_argument("--calibration", default=aruco.CALIBRATION_FILE)
    parser.add_argument("--forbidden-id", type=int)
    parser.add_argument("--csv", default="aruco_hand_check.csv")
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=8001)
    parser.add_argument("--preview-quality", type=int, default=80)
    parser.add_argument("--preview-jpeg", default="aruco_hand_check_preview.jpg")
    parser.add_argument("--preview-jpeg-interval", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
