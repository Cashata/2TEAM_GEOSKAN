#!/usr/bin/env python3
"""Camera adapters for local OpenCV and Pioneer-SDK2."""

from __future__ import annotations

import cv2
import numpy as np


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


class VideoFileCamera:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video file: {}".format(video_path))
        self.finished = False

    def read(self) -> np.ndarray | None:
        if self.finished:
            return None
        ok, frame = self.cap.read()
        if ok:
            return frame
        self.finished = True
        return None

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
