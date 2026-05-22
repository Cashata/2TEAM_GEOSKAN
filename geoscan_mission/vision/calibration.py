#!/usr/bin/env python3
"""Camera calibration helpers for OpenCV and Pioneer camera frames."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_BOARD_SIZE = (6, 9)


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int] | None = None
    rms: float | None = None
    reprojection_error: float | None = None
    board_size: tuple[int, int] | None = None
    square_size: float | None = None
    valid_frames: int | None = None
    total_frames: int | None = None
    alpha: float = 0.0
    _map_cache: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return frame_bgr

        height, width = frame_bgr.shape[:2]
        key = (width, height, float(self.alpha))
        maps = self._map_cache.get(key)
        if maps is None:
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (width, height),
                self.alpha,
                (width, height),
            )
            maps = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                new_camera_matrix,
                (width, height),
                cv2.CV_16SC2,
            )
            self._map_cache[key] = maps

        map1, map2 = maps
        return cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR)


def get_images_from_folder(folder_name: str, file_type: str = "*.jpg") -> list[np.ndarray]:
    """Load images from a directory using a glob pattern."""

    images: list[np.ndarray] = []
    pattern = str(Path(folder_name) / file_type)
    for filename in sorted(glob.glob(pattern)):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    return images


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _object_points(board_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def _mean_reprojection_error(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: Iterable[np.ndarray],
    tvecs: Iterable[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    count = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        total_error += cv2.norm(imgp, projected, cv2.NORM_L2) / max(len(projected), 1)
        count += 1
    return float(total_error / max(count, 1))


def calibrate_camera(
    images: Iterable[np.ndarray],
    board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_size: float = 1.0,
    show: bool = False,
    debug_dir: str | None = None,
) -> CameraCalibration:
    """Estimate camera matrix and distortion from chessboard frames."""

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    objp = _object_points(board_size, square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    total_frames = 0
    out_dir = Path(debug_dir) if debug_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images, 1):
        if image is None:
            continue

        total_frames += 1
        gray = _as_gray(image)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError(
                "all calibration frames must have the same size: {} != {}".format(
                    current_size,
                    image_size,
                )
            )

        found, corners = cv2.findChessboardCorners(gray, board_size, flags)
        annotated = image.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(annotated, board_size, corners2, found)

        if out_dir is not None:
            status = "ok" if found else "miss"
            cv2.imwrite(str(out_dir / "calib_{:04d}_{}.jpg".format(index, status)), annotated)

        if show:
            cv2.imshow("calibration", annotated)
            key = cv2.waitKey(300)
            if key == 27 or key == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    if image_size is None or total_frames == 0:
        raise RuntimeError("no calibration images were loaded")
    if not objpoints:
        raise RuntimeError(
            "no chessboard detections found; check --board-cols/--board-rows and the printed board"
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    reprojection_error = _mean_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms=float(rms),
        reprojection_error=reprojection_error,
        board_size=board_size,
        square_size=square_size,
        valid_frames=len(objpoints),
        total_frames=total_frames,
    )


def save_camera_calibration(calibration: CameraCalibration, path: str) -> None:
    """Save calibration in OpenCV YAML format."""

    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cv_file = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    if not cv_file.isOpened():
        raise RuntimeError("cannot open calibration file for writing: {}".format(path))

    try:
        cv_file.write("mtx", calibration.camera_matrix)
        cv_file.write("dist", calibration.dist_coeffs)
        cv_file.write("camera_matrix", calibration.camera_matrix)
        cv_file.write("dist_coeffs", calibration.dist_coeffs)
        if calibration.image_size is not None:
            cv_file.write("image_width", int(calibration.image_size[0]))
            cv_file.write("image_height", int(calibration.image_size[1]))
        if calibration.rms is not None:
            cv_file.write("rms", float(calibration.rms))
        if calibration.reprojection_error is not None:
            cv_file.write("reprojection_error", float(calibration.reprojection_error))
        if calibration.board_size is not None:
            cv_file.write("board_cols", int(calibration.board_size[0]))
            cv_file.write("board_rows", int(calibration.board_size[1]))
        if calibration.square_size is not None:
            cv_file.write("square_size", float(calibration.square_size))
        if calibration.valid_frames is not None:
            cv_file.write("valid_frames", int(calibration.valid_frames))
        if calibration.total_frames is not None:
            cv_file.write("total_frames", int(calibration.total_frames))
    finally:
        cv_file.release()


def _read_matrix(cv_file, *names: str) -> np.ndarray | None:
    for name in names:
        node = cv_file.getNode(name)
        if not node.empty():
            matrix = node.mat()
            if matrix is not None:
                return matrix
    return None


def _read_int(cv_file, name: str) -> int | None:
    node = cv_file.getNode(name)
    if node.empty():
        return None
    return int(round(node.real()))


def _read_float(cv_file, name: str) -> float | None:
    node = cv_file.getNode(name)
    if node.empty():
        return None
    return float(node.real())


def load_camera_calibration(path: str, alpha: float = 0.0) -> CameraCalibration:
    """Load calibration saved by this module or the original hackathon script."""

    cv_file = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not cv_file.isOpened():
        raise FileNotFoundError("cannot open calibration file: {}".format(path))

    try:
        camera_matrix = _read_matrix(cv_file, "mtx", "camera_matrix")
        dist_coeffs = _read_matrix(cv_file, "dist", "dist_coeffs")
        if camera_matrix is None or dist_coeffs is None:
            raise RuntimeError("calibration file must contain mtx/dist or camera_matrix/dist_coeffs")

        width = _read_int(cv_file, "image_width")
        height = _read_int(cv_file, "image_height")
        image_size = (width, height) if width is not None and height is not None else None
        board_cols = _read_int(cv_file, "board_cols")
        board_rows = _read_int(cv_file, "board_rows")
        board_size = (board_cols, board_rows) if board_cols is not None and board_rows is not None else None

        return CameraCalibration(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            rms=_read_float(cv_file, "rms"),
            reprojection_error=_read_float(cv_file, "reprojection_error"),
            board_size=board_size,
            square_size=_read_float(cv_file, "square_size"),
            valid_frames=_read_int(cv_file, "valid_frames"),
            total_frames=_read_int(cv_file, "total_frames"),
            alpha=alpha,
        )
    finally:
        cv_file.release()


def calibrate(images: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper matching the original calibration.py API."""

    calibration = calibrate_camera(images)
    return calibration.camera_matrix, calibration.dist_coeffs


def save_coefficients(mtx: np.ndarray, dist: np.ndarray, path: str) -> None:
    """Compatibility wrapper matching the original calibration.py API."""

    save_camera_calibration(CameraCalibration(camera_matrix=mtx, dist_coeffs=dist), path)


def load_coefficients(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper matching the original calibration.py API."""

    calibration = load_camera_calibration(path)
    return calibration.camera_matrix, calibration.dist_coeffs
