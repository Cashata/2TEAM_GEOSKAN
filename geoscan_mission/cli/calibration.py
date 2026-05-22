#!/usr/bin/env python3
"""CLI for headless Mini 2 camera calibration."""

from __future__ import annotations

import argparse
import threading
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from geoscan_mission.flight.camera import OpenCvCamera, Sdk2Camera
from geoscan_mission.flight.control import import_pioneer_sdk2
from geoscan_mission.vision.calibration import (
    DEFAULT_BOARD_SIZE,
    CameraCalibration,
    calibrate_camera,
    draw_chessboard_preview,
    find_chessboard_corners,
    get_images_from_folder,
    save_camera_calibration,
)


def create_camera(args: argparse.Namespace):
    if args.camera_source == "opencv":
        return OpenCvCamera(args.camera_index), "opencv:{}".format(args.camera_index)

    sdk2 = import_pioneer_sdk2()
    return Sdk2Camera(sdk2, args.sdk2_camera_type, args.camera_timeout), args.sdk2_camera_type.upper()


def maybe_save_frame(frame: np.ndarray, frames_dir: Path | None, index: int) -> None:
    if frames_dir is None:
        return

    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / "frame_{:04d}.jpg".format(index)
    if not cv2.imwrite(str(path), frame):
        print("WARNING: failed to save {}".format(path), file=sys.stderr)


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


def start_preview_server(
    host: str,
    port: int,
    store: PreviewFrameStore,
) -> tuple[ThreadingHTTPServer, threading.Thread] | None:
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                html = (
                    "<!doctype html><html><head><title>Calibration preview</title>"
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

    try:
        server = ThreadingHTTPServer((host, port), PreviewHandler)
    except OSError as exc:
        print("WARNING: preview HTTP server disabled: {}".format(exc), file=sys.stderr)
        return None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = "http://{}:{}/".format(host, port)
    if host in ("0.0.0.0", "::"):
        address = "http://<drone-ip>:{}/".format(port)
    print("Live calibration preview: {}".format(address))
    return server, thread


def save_preview_jpeg(path: str | None, frame: np.ndarray, last_write: float, interval: float) -> float:
    if not path:
        return last_write

    now = time.monotonic()
    if now - last_write < interval:
        return last_write

    if not cv2.imwrite(path, frame):
        print("WARNING: failed to update preview JPEG: {}".format(path), file=sys.stderr)
    return now


def capture_frames(args: argparse.Namespace) -> list[np.ndarray]:
    camera, camera_name = create_camera(args)
    frames_dir = None if args.no_save_frames else Path(args.frames_dir)
    frames: list[np.ndarray] = []
    board_size = (args.board_cols, args.board_rows)
    preview_store = PreviewFrameStore()
    preview_server = None
    next_capture_time = 0.0
    last_status_time = 0.0
    last_preview_jpeg_time = 0.0
    observed_frames = 0

    try:
        if args.preview_port > 0:
            preview_server = start_preview_server(args.preview_host, args.preview_port, preview_store)

        print("Camera source: {}".format(camera_name))
        if args.warmup_frames > 0:
            print("Skipping {} warmup frames...".format(args.warmup_frames))
        for _ in range(args.warmup_frames):
            camera.read()
            time.sleep(0.05)

        print(
            "Capturing {} valid chessboard frames every {:.2f}s. Move/tilt the checkerboard between accepted frames.".format(
                args.max_frames,
                args.capture_interval,
            )
        )
        while len(frames) < args.max_frames:
            frame = camera.read()
            if frame is None:
                print("WARNING: empty camera frame", file=sys.stderr)
                time.sleep(0.1)
                continue

            observed_frames += 1
            found, corners = find_chessboard_corners(frame, board_size)
            status = "{} valid={}/{} seen={}".format(
                "FOUND" if found else "MISSING",
                len(frames),
                args.max_frames,
                observed_frames,
            )
            annotated = draw_chessboard_preview(frame, board_size, found, corners, status)
            if preview_server is not None:
                preview_store.update(annotated, args.preview_quality)
            last_preview_jpeg_time = save_preview_jpeg(
                args.preview_jpeg,
                annotated,
                last_preview_jpeg_time,
                args.preview_jpeg_interval,
            )

            if args.show:
                cv2.imshow("calibration live preview", annotated)
                key = cv2.waitKey(1)
                if key == 27 or key == ord("q"):
                    print("Preview stopped by keypress")
                    break

            now = time.monotonic()
            if found and now >= next_capture_time:
                frame_index = len(frames) + 1
                frames.append(frame.copy())
                maybe_save_frame(frame, frames_dir, frame_index)
                print("Accepted chessboard frame {}/{}".format(frame_index, args.max_frames))
                next_capture_time = now + args.capture_interval
                continue

            if args.status_interval > 0 and now - last_status_time >= args.status_interval:
                print(status)
                last_status_time = now

            time.sleep(0.03)

    finally:
        preview_store.stop()
        if preview_server is not None:
            preview_server[0].shutdown()
            preview_server[0].server_close()
        if args.show:
            cv2.destroyAllWindows()
        camera.close()

    return frames


def print_summary(calibration: CameraCalibration, output_path: str) -> None:
    print("Calibration saved: {}".format(output_path))
    print("Valid chessboard frames: {}/{}".format(calibration.valid_frames, calibration.total_frames))
    if calibration.image_size is not None:
        print("Image size: {}x{}".format(calibration.image_size[0], calibration.image_size[1]))
    if calibration.rms is not None:
        print("RMS: {:.6f}".format(calibration.rms))
    if calibration.reprojection_error is not None:
        print("Mean reprojection error: {:.6f}".format(calibration.reprojection_error))
    print("Camera matrix:")
    print(calibration.camera_matrix)
    print("Distortion coefficients:")
    print(calibration.dist_coeffs)


def run(args: argparse.Namespace) -> int:
    if args.images:
        images = get_images_from_folder(args.images, args.glob)
        print("Loaded {} images from {}".format(len(images), args.images))
    else:
        images = capture_frames(args)

    calibration = calibrate_camera(
        images,
        board_size=(args.board_cols, args.board_rows),
        square_size=args.square_size,
        show=args.show and bool(args.images),
        debug_dir=args.debug_dir,
    )
    if calibration.valid_frames is not None and calibration.valid_frames < args.min_valid_frames:
        raise RuntimeError(
            "only {} valid chessboard frames found, need at least {}; "
            "collect more varied board angles or lower --min-valid-frames".format(
                calibration.valid_frames,
                args.min_valid_frames,
            )
        )
    save_camera_calibration(calibration, args.output)
    print_summary(calibration, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from chessboard images or a Mini 2 SDK2 camera.",
    )
    parser.add_argument("--images", help="Folder with calibration images. If omitted, frames are captured from a camera.")
    parser.add_argument("--glob", default="*.jpg", help="Glob pattern inside --images.")
    parser.add_argument("--camera-source", choices=["sdk2", "opencv"], default="sdk2")
    parser.add_argument("--sdk2-camera-type", default="OPT")
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=25)
    parser.add_argument("--capture-interval", type=float, default=1.0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--frames-dir", default="calibration_frames")
    parser.add_argument("--no-save-frames", action="store_true", help="Do not save captured source frames.")
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=8000, help="HTTP live preview port. Use 0 to disable.")
    parser.add_argument("--preview-quality", type=int, default=80, help="JPEG quality for HTTP preview.")
    parser.add_argument("--preview-jpeg", default="calibration_preview.jpg", help="Continuously updated preview JPEG path. Empty string disables it.")
    parser.add_argument("--preview-jpeg-interval", type=float, default=0.2)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--output", default="data.yml")
    parser.add_argument("--debug-dir", help="Save annotated chessboard detections here.")
    parser.add_argument("--min-valid-frames", type=int, default=8)
    parser.add_argument("--board-cols", type=int, default=DEFAULT_BOARD_SIZE[0], help="Inner chessboard corners by columns.")
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD_SIZE[1], help="Inner chessboard corners by rows.")
    parser.add_argument("--square-size", type=float, default=1.0, help="Physical square size in any consistent unit.")
    parser.add_argument("--show", action="store_true", help="Show frames with detected corners. Local desktop only.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.board_cols < 2 or args.board_rows < 2:
        raise ValueError("--board-cols and --board-rows must be at least 2")
    if args.square_size <= 0:
        raise ValueError("--square-size must be positive")
    if args.camera_timeout <= 0:
        raise ValueError("--camera-timeout must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.capture_interval < 0:
        raise ValueError("--capture-interval must be >= 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.min_valid_frames <= 0:
        raise ValueError("--min-valid-frames must be positive")
    if args.preview_port < 0 or args.preview_port > 65535:
        raise ValueError("--preview-port must be between 0 and 65535")
    if not (1 <= args.preview_quality <= 100):
        raise ValueError("--preview-quality must be between 1 and 100")
    if args.preview_jpeg_interval < 0:
        raise ValueError("--preview-jpeg-interval must be >= 0")
    if args.status_interval < 0:
        raise ValueError("--status-interval must be >= 0")
    if not args.sdk2_camera_type.strip():
        raise ValueError("--sdk2-camera-type must not be empty")
    if args.images and not Path(args.images).exists():
        raise ValueError("--images folder does not exist: {}".format(args.images))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        return run(args)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
