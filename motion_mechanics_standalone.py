#!/usr/bin/env python3
"""Самодостаточный исполнитель готовой траектории для Pioneer Mini.

Этот файл больше не строит заранее заданный квадрат и не выбирает маршрут сам.
Он получает уже готовые точки траектории, например spline из другого модуля,
и только исполняет механику движения: постоянная скорость вперед в body frame,
боковая P-компенсация отклонения от пути, высота по P и yaw как
feed-forward по касательной + P по ошибке курса.

Основной способ использовать из другого кода:
    run_motion([(x1, y1, z1), (x2, y2, z2), ...])

Если файл запускается напрямую, точки загружаются из TRAJECTORY_FILE. JSON может
быть списком [[x, y, z], ...] или объектом {"points": [[x, y, z], ...]}.
CSV может быть с колонками x,y,z или просто строками из трех чисел.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence


TRAJECTORY_FILE = "trajectory_points.json"

SPEED = 0.08
KP_LATERAL = 1.2
KP_Z = 1.0
KP_YAW = 1.6
MAX_XY_SPEED = SPEED
MAX_Z_SPEED = SPEED
MAX_YAW_RATE = 0.7
FINISH_TOLERANCE = 0.12
CONTROL_INTERVAL = 0.05
COMMAND_INTERVAL = 0.1
TAKEOFF_WAIT = 2.0


Point3 = tuple[float, float, float]


# Ограничивает значение диапазоном, чтобы команды скорости не выходили за лимиты.
def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# Нормализует угол в радианах в диапазон [-pi, pi] для кратчайшей ошибки yaw.
def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# Считает 3D-расстояние между текущей позицией и точкой траектории.
def distance_3d(a: Point3, b: Point3) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


# Приводит входной список/tuple к валидной точке формата (x, y, z).
def parse_point(value: Sequence[object]) -> Point3:
    if len(value) < 3:
        raise ValueError("trajectory point must contain x, y, z")
    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(item) for item in point):
        raise ValueError("trajectory point values must be finite")
    return point


# Проверяет готовую траекторию и возвращает список точек в нужном типе.
def normalize_points(points: Iterable[Sequence[object]]) -> list[Point3]:
    normalized = [parse_point(point) for point in points]
    if len(normalized) < 2:
        raise ValueError("trajectory must contain at least two points")
    return normalized


# Загружает точки из JSON: [[x, y, z], ...] или {"points": [[x, y, z], ...]}.
def load_json_points(path: Path) -> list[Point3]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("points")
    if not isinstance(data, list):
        raise ValueError("JSON trajectory must be a list or an object with points")
    return normalize_points(data)


# Загружает точки из CSV с колонками x,y,z или из строк вида x,y,z.
def load_csv_points(path: Path) -> list[Point3]:
    with path.open(newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(f)
            return normalize_points((row["x"], row["y"], row["z"]) for row in reader)
        reader = csv.reader(f)
        return normalize_points(row for row in reader if row)


# Загружает готовую траекторию из внешнего JSON/CSV-файла.
def load_trajectory_points(path: str = TRAJECTORY_FILE) -> list[Point3]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError("trajectory file not found: {}".format(source))
    suffix = source.suffix.lower()
    if suffix == ".json":
        return load_json_points(source)
    if suffix == ".csv":
        return load_csv_points(source)
    raise ValueError("trajectory file must be .json or .csv")


# Ищет ближайшую к дрону точку только вперед от текущего индекса прогресса.
def nearest_future_index(points: list[Point3], position: Point3, start_index: int) -> int:
    best_index = min(max(start_index, 0), len(points) - 1)
    best_distance = float("inf")
    px, py, pz = position
    for index in range(best_index, len(points)):
        point = points[index]
        distance = (point[0] - px) ** 2 + (point[1] - py) ** 2 + (point[2] - pz) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


# Считает целевой yaw как направление касательной к уже готовой траектории.
def tangent_yaw(points: list[Point3], index: int) -> float:
    index = min(max(index, 0), len(points) - 1)
    before = points[max(0, index - 1)]
    after = points[min(len(points) - 1, index + 1)]
    dx = after[0] - before[0]
    dy = after[1] - before[1]
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return 0.0
    return math.atan2(dy, dx)


# Считает feed-forward yaw-rate по изменению касательной вдоль траектории.
def yaw_feed_forward(points: list[Point3], index: int, speed: float) -> float:
    prev_i = max(0, index - 1)
    next_i = min(len(points) - 1, index + 1)
    ds = math.hypot(points[next_i][0] - points[prev_i][0], points[next_i][1] - points[prev_i][1])
    if ds <= 1e-6:
        return 0.0
    return normalize_angle_rad(tangent_yaw(points, next_i) - tangent_yaw(points, prev_i)) * speed / ds


# Читает текущую локальную позицию дрона из LPS.
def get_position(drone) -> Point3:
    position = drone.get_local_position_lps()
    if position is None or len(position) < 3:
        raise RuntimeError("get_local_position_lps() returned no valid position")
    return float(position[0]), float(position[1]), float(position[2])


# Читает текущий yaw дрона из LPS и переводит градусы в радианы.
def get_yaw_rad(drone) -> float:
    yaw_deg = drone.get_local_yaw_lps()
    if yaw_deg is None:
        raise RuntimeError("get_local_yaw_lps() returned no valid yaw")
    return math.radians(float(yaw_deg))


# Отправляет body-fixed speed-команду в SDK2.
def send_speed(drone, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
    try:
        result = drone.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate, interval=COMMAND_INTERVAL)
    except TypeError:
        result = drone.set_manual_speed_body_fixed(vx, vy, vz, yaw_rate)
    if result is False:
        raise RuntimeError("set_manual_speed_body_fixed() returned False")


# Считает одну команду движения по ближайшей точке готовой траектории.
def compute_follow_command(drone, points: list[Point3], progress_index: int, speed: float) -> tuple[tuple[float, float, float, float], int, bool]:
    position = get_position(drone)
    yaw = get_yaw_rad(drone)
    target_i = nearest_future_index(points, position, progress_index)
    target = points[target_i]
    complete = target_i >= len(points) - 2 and distance_3d(position, points[-1]) <= FINISH_TOLERANCE
    if complete:
        return (0.0, 0.0, 0.0, 0.0), target_i, True

    yaw_target = tangent_yaw(points, target_i)
    to_path_x = target[0] - position[0]
    to_path_y = target[1] - position[1]
    lateral_error = -to_path_x * math.sin(yaw_target) + to_path_y * math.cos(yaw_target)
    z_error = target[2] - position[2]
    yaw_error = normalize_angle_rad(yaw_target - yaw)

    vx_body = clip(speed, 0.0, MAX_XY_SPEED)
    vy_body = clip(KP_LATERAL * lateral_error, -MAX_XY_SPEED, MAX_XY_SPEED)
    vz = clip(KP_Z * z_error, -MAX_Z_SPEED, MAX_Z_SPEED)
    yaw_rate = clip(yaw_feed_forward(points, target_i, speed) + KP_YAW * yaw_error, -MAX_YAW_RATE, MAX_YAW_RATE)
    return (vx_body, vy_body, vz, yaw_rate), target_i, False


# Исполняет уже готовую траекторию до ее конца.
def follow_trajectory(drone, points: list[Point3], speed: float = SPEED) -> bool:
    progress = 0
    while True:
        command, target_i, complete = compute_follow_command(drone, points, progress, speed)
        progress = max(progress, target_i)
        send_speed(drone, *command)
        if complete:
            return True
        time.sleep(CONTROL_INTERVAL)


# Открывает Pioneer SDK2 и создает объект дрона.
def create_drone():
    sdk2 = importlib.import_module("pioneer_sdk2")
    return sdk2.Pioneer(wait_callback=True, safety_command=True)


# Внешняя функция: принимает уже готовый spline/список точек и выполняет полет.
def run_motion(points: Iterable[Sequence[object]], drone=None, speed: float = SPEED, takeoff: bool = True, land: bool = True) -> bool:
    trajectory = normalize_points(points)
    drone = drone or create_drone()
    try:
        if takeoff:
            if hasattr(drone, "arm"):
                drone.arm(timeout=5, retries=1)
            drone.takeoff()
            time.sleep(TAKEOFF_WAIT)
        return follow_trajectory(drone, trajectory, speed=speed)
    finally:
        try:
            send_speed(drone, 0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        if land:
            drone.land()
            if hasattr(drone, "disarm"):
                time.sleep(1.0)
                drone.disarm()


# Точка входа: при прямом запуске берет точки из TRAJECTORY_FILE и исполняет их.
def main() -> int:
    points = load_trajectory_points(TRAJECTORY_FILE)
    run_motion(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
