#!/usr/bin/env python3
"""Самодостаточное построение маршрута без полета и внешних модулей проекта.

Файл только строит координаты маршрута. Он не открывает камеру, не подключается к
дрону, не отправляет команды скорости и ничего не выводит в консоль.

Основной вход для внешнего кода:
    build_route(...)
    build_spline_from_waypoints(...)

Координаты возвращаются в локальной системе дрона как список точек:
    [(x_m, y_m, z_m), ...]
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


Point3 = tuple[float, float, float]

DEFAULT_WAYPOINTS: list[Point3] = [
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.5),
    (0.0, 0.0, 1.0),
]


# Приводит входную точку к формату (x, y, z) и проверяет, что значения конечные.
def parse_point(value: Sequence[object]) -> Point3:
    if len(value) < 3:
        raise ValueError("point must contain x, y, z")
    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(item) for item in point):
        raise ValueError("point values must be finite")
    return point


# Приводит любой список входных координат к списку Point3.
def normalize_points(points: Iterable[Sequence[object]]) -> list[Point3]:
    normalized = [parse_point(point) for point in points]
    if not normalized:
        raise ValueError("route must contain at least one point")
    return normalized


# Строит равномерную сетку чисел от start до stop включительно.
def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


# Считает расстояние между двумя 3D-точками.
def distance_3d(a: Point3, b: Point3) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


# Линейно интерполирует точку между a и b.
def interpolate_point(a: Point3, b: Point3, ratio: float) -> Point3:
    return (
        a[0] + (b[0] - a[0]) * ratio,
        a[1] + (b[1] - a[1]) * ratio,
        a[2] + (b[2] - a[2]) * ratio,
    )


# Пересэмплирует ломаную так, чтобы получить ровно count точек по длине пути.
def resample_polyline(points: Sequence[Point3], count: int) -> list[Point3]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not points:
        raise ValueError("trajectory must contain at least one point")
    if count == 1:
        return [parse_point(points[0])]
    if len(points) == 1:
        point = parse_point(points[0])
        return [point for _ in range(count)]

    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + distance_3d(points[index - 1], points[index]))

    total_distance = distances[-1]
    if total_distance <= 0:
        point = parse_point(points[0])
        return [point for _ in range(count)]

    result: list[Point3] = []
    segment_index = 0
    for sample_index in range(count):
        target_distance = total_distance * sample_index / (count - 1)
        while segment_index < len(distances) - 2 and distances[segment_index + 1] < target_distance:
            segment_index += 1

        segment_start = distances[segment_index]
        segment_end = distances[segment_index + 1]
        ratio = 0.0 if segment_end <= segment_start else (target_distance - segment_start) / (segment_end - segment_start)
        result.append(interpolate_point(points[segment_index], points[segment_index + 1], ratio))
    return result


# Проверяет, совпадают ли две точки с учетом маленькой погрешности.
def is_same_point(a: Point3, b: Point3, tolerance: float = 1e-9) -> bool:
    return distance_3d(a, b) <= tolerance


# Считает производные в waypoint для Catmull-Rom/Hermite spline.
def catmull_rom_derivatives(points: Sequence[Point3], parameters: Sequence[float]) -> list[Point3]:
    derivatives: list[Point3] = []
    last_index = len(points) - 1
    closed = len(points) > 2 and is_same_point(points[0], points[-1])

    for index, point in enumerate(points):
        if closed and index in (0, last_index):
            previous_index = last_index - 1
            next_index = 1
            denominator = distance_3d(points[previous_index], point) + distance_3d(point, points[next_index])
            if denominator <= 0:
                derivatives.append((0.0, 0.0, 0.0))
            else:
                derivatives.append(
                    (
                        (points[next_index][0] - points[previous_index][0]) / denominator,
                        (points[next_index][1] - points[previous_index][1]) / denominator,
                        (points[next_index][2] - points[previous_index][2]) / denominator,
                    )
                )
        elif index == 0:
            denominator = parameters[1] - parameters[0]
            derivatives.append(
                (
                    (points[1][0] - point[0]) / denominator,
                    (points[1][1] - point[1]) / denominator,
                    (points[1][2] - point[2]) / denominator,
                )
            )
        elif index == last_index:
            denominator = parameters[last_index] - parameters[last_index - 1]
            derivatives.append(
                (
                    (point[0] - points[last_index - 1][0]) / denominator,
                    (point[1] - points[last_index - 1][1]) / denominator,
                    (point[2] - points[last_index - 1][2]) / denominator,
                )
            )
        else:
            denominator = parameters[index + 1] - parameters[index - 1]
            derivatives.append(
                (
                    (points[index + 1][0] - points[index - 1][0]) / denominator,
                    (points[index + 1][1] - points[index - 1][1]) / denominator,
                    (points[index + 1][2] - points[index - 1][2]) / denominator,
                )
            )
    return derivatives


# Вычисляет одну точку Hermite-сегмента spline.
def evaluate_hermite(p0: Point3, p1: Point3, m0: Point3, m1: Point3, segment_length: float, u: float) -> Point3:
    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    return (
        h00 * p0[0] + h10 * segment_length * m0[0] + h01 * p1[0] + h11 * segment_length * m1[0],
        h00 * p0[1] + h10 * segment_length * m0[1] + h01 * p1[1] + h11 * segment_length * m1[1],
        h00 * p0[2] + h10 * segment_length * m0[2] + h01 * p1[2] + h11 * segment_length * m1[2],
    )


# Строит spline из готовых waypoint и возвращает ровно num_points точек.
def build_spline_from_waypoints(waypoints: Iterable[Sequence[object]], num_points: int = 5000) -> list[Point3]:
    points = normalize_points(waypoints)
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(points) < 2:
        raise ValueError("trajectory requires at least two waypoints")
    if len(points) == 2:
        return resample_polyline(points, num_points)

    parameters = [0.0]
    for index in range(1, len(points)):
        segment_length = distance_3d(points[index - 1], points[index])
        parameters.append(parameters[-1] + max(segment_length, 1e-6))

    derivatives = catmull_rom_derivatives(points, parameters)
    dense_count = max(num_points * 2, len(points) * 32)
    samples_per_segment = max(8, dense_count // (len(points) - 1))
    dense_points: list[Point3] = []

    for segment_index in range(len(points) - 1):
        segment_length = parameters[segment_index + 1] - parameters[segment_index]
        start_sample = 0 if segment_index == 0 else 1
        for sample_index in range(start_sample, samples_per_segment + 1):
            u = sample_index / samples_per_segment
            dense_points.append(
                evaluate_hermite(
                    points[segment_index],
                    points[segment_index + 1],
                    derivatives[segment_index],
                    derivatives[segment_index + 1],
                    segment_length,
                    u,
                )
            )

    return resample_polyline(dense_points, num_points)


# Строит грубые waypoint для прохода змейкой по квадратной зоне.
def build_lawnmower_waypoints(area_size: float, margin: float, grid_size: int, height: float) -> list[Point3]:
    if area_size <= 0:
        raise ValueError("area_size must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if margin < 0 or margin * 2 >= area_size:
        raise ValueError("margin must be non-negative and smaller than half of area_size")

    lo = -area_size / 2.0 + margin
    hi = area_size / 2.0 - margin
    xs = linspace(lo, hi, grid_size)
    ys = linspace(lo, hi, grid_size)

    points: list[Point3] = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((round(x, 3), round(y, 3), float(height)))
    return points


# Строит грубые waypoint квадратного маршрута с возвратом в начальную точку.
def build_square_waypoints(area_size: float, margin: float, height: float) -> list[Point3]:
    if area_size <= 0:
        raise ValueError("area_size must be positive")
    if margin < 0 or margin * 2 >= area_size:
        raise ValueError("margin must be non-negative and smaller than half of area_size")

    lo = -area_size / 2.0 + margin
    hi = area_size / 2.0 - margin
    return [
        (lo, lo, float(height)),
        (hi, lo, float(height)),
        (hi, hi, float(height)),
        (lo, hi, float(height)),
        (lo, lo, float(height)),
    ]


# Строит грубые waypoint для нескольких высот, проходя каждую высоту змейкой.
def build_cube_waypoints(
    area_size: float,
    margin: float,
    grid_size: int,
    heights: Sequence[float],
) -> list[Point3]:
    if not heights:
        raise ValueError("heights must contain at least one value")

    points: list[Point3] = []
    for layer_index, height in enumerate(heights):
        layer = build_lawnmower_waypoints(area_size, margin, grid_size, float(height))
        if layer_index % 2:
            layer = list(reversed(layer))
        points.extend(layer)
    return points


# Выбирает тип грубого маршрута или принимает уже готовые waypoint.
def build_waypoints(
    pattern: str = "waypoints",
    waypoints: Iterable[Sequence[object]] | None = None,
    area_size: float = 0.6,
    margin: float = 0.2,
    grid_size: int = 4,
    height: float = 0.6,
    high_height: float = 1.0,
    layers: Sequence[float] | None = None,
) -> list[Point3]:
    if waypoints is not None:
        return normalize_points(waypoints)

    if pattern == "waypoints":
        return list(DEFAULT_WAYPOINTS)
    if pattern == "square":
        return build_square_waypoints(area_size, margin, height)
    if pattern == "lawnmower":
        return build_lawnmower_waypoints(area_size, margin, grid_size, height)
    if pattern == "cube":
        return build_cube_waypoints(area_size, margin, grid_size, layers or (height, high_height))
    raise ValueError("unknown route pattern: {}".format(pattern))


# Строит итоговый маршрут: сначала waypoint, затем при необходимости spline из num_points точек.
def build_route(
    pattern: str = "square",
    waypoints: Iterable[Sequence[object]] | None = None,
    num_points: int = 5000,
    area_size: float = 0.6,
    margin: float = 0.2,
    grid_size: int = 4,
    height: float = 0.6,
    high_height: float = 1.0,
    layers: Sequence[float] | None = None,
    use_spline: bool = True,
) -> list[Point3]:
    route_waypoints = build_waypoints(
        pattern=pattern,
        waypoints=waypoints,
        area_size=area_size,
        margin=margin,
        grid_size=grid_size,
        height=height,
        high_height=high_height,
        layers=layers,
    )
    if not use_spline:
        return route_waypoints
    return build_spline_from_waypoints(route_waypoints, num_points)
