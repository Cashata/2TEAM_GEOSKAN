#!/usr/bin/env python3
"""Самодостаточный трекер ArUco-маркеров для Pioneer Mini.

Файл не зависит от модулей geoscan_mission. Он открывает камеру, ищет
ArUco-маркеры и передает данные о каждом найденном маркере в emit_marker().
По умолчанию emit_marker() обновляет LATEST_MARKERS и пишет записи в CSV.
Если координаты нужно передавать в другой модуль, задайте MARKER_CALLBACK.
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np


CAMERA_SOURCE = "sdk2"
SDK2_CAMERA_TYPE = "OPT"
CAMERA_INDEX = 0
CAMERA_TIMEOUT = 2.0

ARUCO_DICTIONARY = "DICT_4X4_1000"
TARGETS = {
    4: {"letter": "B"},
    15: {"letter": "A"},
    10: {"letter": "N"},
    16: {"letter": "D"},
}
WORD_ORDER = (4, 15, 10, 16)
DELIVERY_MARKER_ID = 15
FORBIDDEN_MARKER_ID: int | None = None

CALIBRATION_FILE = "data.yml"
SMALL_MARKER_SIZE_M = 0.07
LARGE_MARKER_SIZE_M = 0.15
MARKER_SIZES_M = {
    4: SMALL_MARKER_SIZE_M,
    15: LARGE_MARKER_SIZE_M,
    10: SMALL_MARKER_SIZE_M,
    16: SMALL_MARKER_SIZE_M,
}
CSV_PATH = "aruco_tracking.csv"
FRAME_LIMIT = 0

MarkerRecord = dict[str, object]
MarkerCallback = Callable[[MarkerRecord], None]

MARKER_CALLBACK: MarkerCallback | None = None
LATEST_MARKERS: dict[int, MarkerRecord] = {}


def set_forbidden_marker(marker_id: int | None) -> None:
    global FORBIDDEN_MARKER_ID
    FORBIDDEN_MARKER_ID = None if marker_id is None else int(marker_id)


def target_letter(marker_id: int) -> str:
    target = TARGETS.get(marker_id)
    return "" if target is None else str(target["letter"])


def target_type(marker_id: int) -> str:
    if marker_id not in TARGETS:
        return "unknown"
    if FORBIDDEN_MARKER_ID is not None and marker_id == FORBIDDEN_MARKER_ID:
        return "forbidden"
    return "allowed"


def allowed_target_ids() -> list[int]:
    return [marker_id for marker_id in WORD_ORDER if marker_id in TARGETS and target_type(marker_id) == "allowed"]


def known_found_ids() -> list[int]:
    return [marker_id for marker_id in WORD_ORDER if marker_id in LATEST_MARKERS]


def allowed_found_ids() -> list[int]:
    return [marker_id for marker_id in allowed_target_ids() if marker_id in LATEST_MARKERS]


def forbidden_found_ids() -> list[int]:
    return [marker_id for marker_id in WORD_ORDER if target_type(marker_id) == "forbidden" and marker_id in LATEST_MARKERS]


def current_allowed_word() -> str:
    return "".join(target_letter(marker_id) for marker_id in allowed_found_ids())


def current_detected_word() -> str:
    return "".join(target_letter(marker_id) for marker_id in known_found_ids())


def expected_allowed_word() -> str:
    return "".join(target_letter(marker_id) for marker_id in allowed_target_ids())


def delivery_candidate_id() -> int | None:
    return DELIVERY_MARKER_ID if DELIVERY_MARKER_ID in LATEST_MARKERS else None


def build_task_state() -> MarkerRecord:
    candidate_id = delivery_candidate_id()
    return {
        "assignment_word": "".join(target_letter(marker_id) for marker_id in WORD_ORDER),
        "detected_word": current_detected_word(),
        "expected_allowed_word": expected_allowed_word(),
        "allowed_word": current_allowed_word(),
        "all_markers_found": len(known_found_ids()) == len(WORD_ORDER),
        "word_complete": len(allowed_found_ids()) == len(allowed_target_ids()),
        "forbidden_marker_id": FORBIDDEN_MARKER_ID,
        "known_found_ids": known_found_ids(),
        "allowed_ids": allowed_found_ids(),
        "forbidden_ids": forbidden_found_ids(),
        "delivery_marker_id": DELIVERY_MARKER_ID,
        "delivery_marker_found": DELIVERY_MARKER_ID in LATEST_MARKERS,
        "delivery_marker_allowed": target_type(DELIVERY_MARKER_ID) == "allowed",
        "delivery_candidate_id": candidate_id,
        "delivery_candidate_letter": target_letter(candidate_id) if candidate_id is not None else "",
    }


# Преобразует строковое имя словаря ArUco в предустановленный словарь OpenCV.
def resolve_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without cv2.aruco support")
    normalized = name.strip().upper()
    if not normalized.startswith("DICT_"):
        normalized = "DICT_" + normalized
    if not hasattr(cv2.aruco, normalized):
        available = sorted(item for item in dir(cv2.aruco) if item.startswith("DICT_"))
        raise RuntimeError("Unknown ArUco dictionary {}. Examples: {}".format(name, ", ".join(available[:12])))
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, normalized))


# Создает параметры детектора так, чтобы работали старые и новые версии OpenCV.
def create_detector_parameters():
    try:
        return cv2.aruco.DetectorParameters()
    except AttributeError:
        return cv2.aruco.DetectorParameters_create()


# Создает объект ArUco-детектора, используя новый API OpenCV, если он доступен.
def create_detector(dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


# Открывает SDK2-камеру на Pioneer Mini.
def open_sdk2_camera():
    sdk2 = importlib.import_module("pioneer_sdk2")
    camera_type_name = SDK2_CAMERA_TYPE.upper()
    if not hasattr(sdk2.CameraType, camera_type_name):
        available = sorted(name for name in dir(sdk2.CameraType) if name.isupper())
        raise RuntimeError("CameraType.{} is unavailable. Available: {}".format(camera_type_name, ", ".join(available)))
    camera_type = getattr(sdk2.CameraType, camera_type_name)
    try:
        return sdk2.Camera(camera_type=camera_type)
    except TypeError:
        return sdk2.Camera(camera_type)


# Открывает обычную OpenCV-камеру для локальных тестов без SDK дрона.
def open_opencv_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open OpenCV camera index {}".format(CAMERA_INDEX))
    return cap


# Читает один BGR-кадр из SDK2 Camera или OpenCV VideoCapture.
def read_frame(camera):
    if hasattr(camera, "get_cv_frame"):
        return camera.get_cv_frame(timeout=CAMERA_TIMEOUT)
    ok, frame = camera.read()
    return frame if ok else None


# Закрывает используемый backend камеры.
def close_camera(camera) -> None:
    for method_name in ("close", "stop", "release"):
        method = getattr(camera, method_name, None)
        if callable(method):
            method()
            return


# Загружает OpenCV YAML-калибровку, если файл существует.
def load_calibration(path: str):
    if not path or not Path(path).exists():
        return None, None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None, None
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        return None, None
    return camera_matrix, dist_coeffs


# Находит маркеры в кадре и возвращает стандартный вывод OpenCV: corners/ids.
def detect_markers(frame, dictionary, params, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


# Считает координаты центра и углов найденного маркера в пикселях.
def marker_pixel_coordinates(corner) -> tuple[list[float], list[list[float]]]:
    points = np.asarray(corner[0], dtype=np.float64)
    center = points.mean(axis=0)
    return center.tolist(), points.tolist()


# Возвращает физический размер маркера в метрах по его ID.
def marker_size_for_id(marker_id: int) -> float | None:
    size = MARKER_SIZES_M.get(marker_id)
    if size is None or size <= 0:
        return None
    return float(size)


# Оценивает pose маркера в координатах камеры, если есть калибровка и размер маркера.
def estimate_marker_pose(corner, marker_size_m: float | None, camera_matrix, dist_coeffs):
    if camera_matrix is None or dist_coeffs is None or marker_size_m is None:
        return None
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corner], marker_size_m, camera_matrix, dist_coeffs)
    return rvecs[0, 0].tolist(), tvecs[0, 0].tolist()


# Собирает структурированную запись о маркере для передачи в sink.
def build_marker_record(frame_id: int, marker_id: int, corner, camera_matrix, dist_coeffs, first_seen: bool) -> MarkerRecord:
    center_px, corners_px = marker_pixel_coordinates(corner)
    marker_size_m = marker_size_for_id(marker_id)
    pose = estimate_marker_pose(corner, marker_size_m, camera_matrix, dist_coeffs)
    record: MarkerRecord = {
        "timestamp": time.time(),
        "frame_id": frame_id,
        "id": marker_id,
        "letter": target_letter(marker_id),
        "type": target_type(marker_id),
        "first_seen": first_seen,
        "marker_size_m": marker_size_m,
        "center_px": center_px,
        "corners_px": corners_px,
        "rvec": None,
        "tvec_m": None,
    }
    if pose is not None:
        record["rvec"], record["tvec_m"] = pose
    return record


# Добавляет запись о маркере в CSV, сохраняя вложенные координаты как JSON-строки.
def encode_csv_value(value):
    if value is None or isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def append_csv(path: str, record: MarkerRecord) -> None:
    if not path:
        return
    csv_path = Path(path)
    file_exists = csv_path.exists()
    row = {key: encode_csv_value(value) for key, value in record.items()}
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# Передает найденный маркер наружу: обновляет память, пишет CSV и вызывает callback.
def emit_marker(record: MarkerRecord) -> None:
    marker_id = int(record["id"])
    LATEST_MARKERS[marker_id] = record
    record.update(build_task_state())
    record["delivery_candidate"] = marker_id == record["delivery_candidate_id"]
    append_csv(CSV_PATH, record)
    if MARKER_CALLBACK is not None:
        MARKER_CALLBACK(record)


# Обрабатывает один кадр: находит ArUco и отправляет записи через emit_marker().
def process_frame(frame, frame_id: int, dictionary, params, detector, camera_matrix, dist_coeffs, seen_ids: set[int]) -> int:
    corners, ids, _ = detect_markers(frame, dictionary, params, detector)
    if ids is None:
        return 0
    count = 0
    for index, raw_id in enumerate(ids.flatten()):
        marker_id = int(raw_id)
        first_seen = marker_id not in seen_ids
        seen_ids.add(marker_id)
        record = build_marker_record(frame_id, marker_id, corners[index], camera_matrix, dist_coeffs, first_seen)
        emit_marker(record)
        count += 1
    return count


# Запускает основной цикл трекинга: читает кадры, ищет маркеры и передает координаты.
def track_aruco(camera) -> None:
    dictionary = resolve_aruco_dictionary(ARUCO_DICTIONARY)
    params = create_detector_parameters()
    detector = create_detector(dictionary, params)
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)
    seen_ids: set[int] = set()
    frame_id = 0

    while FRAME_LIMIT <= 0 or frame_id < FRAME_LIMIT:
        frame = read_frame(camera)
        if frame is None:
            time.sleep(0.05)
            continue
        process_frame(frame, frame_id, dictionary, params, detector, camera_matrix, dist_coeffs, seen_ids)
        frame_id += 1


# Точка входа: открывает выбранную камеру, трекает ArUco и закрывает камеру.
def main() -> int:
    camera = open_sdk2_camera() if CAMERA_SOURCE == "sdk2" else open_opencv_camera()
    try:
        track_aruco(camera)
    finally:
        close_camera(camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
