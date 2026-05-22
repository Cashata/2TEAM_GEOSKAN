#!/usr/bin/env python3
"""Reusable waypoint and trajectory pattern builders."""

from __future__ import annotations

import argparse
import math
from typing import Sequence


Point3 = tuple[float, float, float]


DEFAULT_WAYPOINTS = [
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.5),
    (0.0, 0.0, 1.0),
]


def parse_waypoint(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("waypoint must be x,y,z")
    return parts[0], parts[1], parts[2]


def parse_float_list(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("list must contain at least one number")
    return parts


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [(start + stop) / 2.0]
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def distance_3d(a: Point3, b: Point3) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def interpolate_point(a: Point3, b: Point3, ratio: float) -> Point3:
    return (
        a[0] + (b[0] - a[0]) * ratio,
        a[1] + (b[1] - a[1]) * ratio,
        a[2] + (b[2] - a[2]) * ratio,
    )


def resample_polyline(points: Sequence[Point3], count: int) -> list[Point3]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not points:
        raise ValueError("trajectory must contain at least one point")
    if count == 1:
        return [tuple(float(value) for value in points[0])]
    if len(points) == 1:
        point = tuple(float(value) for value in points[0])
        return [point for _ in range(count)]

    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + distance_3d(points[index - 1], points[index]))

    total_distance = distances[-1]
    if total_distance <= 0:
        point = tuple(float(value) for value in points[0])
        return [point for _ in range(count)]

    result: list[Point3] = []
    segment_index = 0
    for sample_index in range(count):
        target_distance = total_distance * sample_index / (count - 1)
        while segment_index < len(distances) - 2 and distances[segment_index + 1] < target_distance:
            segment_index += 1

        segment_start = distances[segment_index]
        segment_end = distances[segment_index + 1]
        if segment_end <= segment_start:
            ratio = 0.0
        else:
            ratio = (target_distance - segment_start) / (segment_end - segment_start)
        result.append(interpolate_point(points[segment_index], points[segment_index + 1], ratio))
    return result


def _is_same_point(a: Point3, b: Point3, tolerance: float = 1e-9) -> bool:
    return distance_3d(a, b) <= tolerance


def _catmull_rom_derivatives(points: Sequence[Point3], parameters: Sequence[float]) -> list[Point3]:
    derivatives: list[Point3] = []
    last_index = len(points) - 1
    closed = len(points) > 2 and _is_same_point(points[0], points[-1])

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


def _evaluate_hermite(p0: Point3, p1: Point3, m0: Point3, m1: Point3, segment_length: float, u: float) -> Point3:
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


def sample_spline_trajectory(waypoints: Sequence[Point3], num_points: int) -> list[Point3]:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(waypoints) < 2:
        raise ValueError("trajectory requires at least two waypoints")
    if len(waypoints) == 2:
        return resample_polyline(waypoints, num_points)

    parameters = [0.0]
    for index in range(1, len(waypoints)):
        segment_length = distance_3d(waypoints[index - 1], waypoints[index])
        parameters.append(parameters[-1] + max(segment_length, 1e-6))

    derivatives = _catmull_rom_derivatives(waypoints, parameters)
    dense_count = max(num_points * 2, len(waypoints) * 32)
    samples_per_segment = max(8, dense_count // (len(waypoints) - 1))
    dense_points: list[Point3] = []

    for segment_index in range(len(waypoints) - 1):
        segment_length = parameters[segment_index + 1] - parameters[segment_index]
        start_sample = 0 if segment_index == 0 else 1
        for sample_index in range(start_sample, samples_per_segment + 1):
            u = sample_index / samples_per_segment
            dense_points.append(
                _evaluate_hermite(
                    waypoints[segment_index],
                    waypoints[segment_index + 1],
                    derivatives[segment_index],
                    derivatives[segment_index + 1],
                    segment_length,
                    u,
                )
            )

    return resample_polyline(dense_points, num_points)


def build_lawnmower_points(area_size: float, margin: float, grid_size: int, height: float) -> list[tuple[float, float, float]]:
    lo = -area_size / 2.0 + margin
    hi = area_size / 2.0 - margin
    xs = linspace(lo, hi, grid_size)
    ys = linspace(lo, hi, grid_size)

    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((round(x, 3), round(y, 3), height))
    return points


def build_square_points(area_size: float, margin: float, height: float) -> list[tuple[float, float, float]]:
    lo = -area_size / 2.0 + margin
    hi = area_size / 2.0 - margin
    return [
        (lo, lo, height),
        (hi, lo, height),
        (hi, hi, height),
        (lo, hi, height),
        (lo, lo, height),
    ]


def resolve_waypoints(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    if args.waypoint:
        return args.waypoint

    if args.trajectory == "waypoints":
        return list(DEFAULT_WAYPOINTS)

    if args.trajectory == "square":
        return build_square_points(args.area_size, args.margin, args.height)

    if args.trajectory == "lawnmower":
        return build_lawnmower_points(args.area_size, args.margin, args.grid_size, args.height)

    if args.trajectory == "cube":
        points = []
        layers = args.layers or [args.height, args.high_height]
        for layer_index, height in enumerate(layers):
            layer = build_lawnmower_points(args.area_size, args.margin, args.grid_size, height)
            if layer_index % 2:
                layer = list(reversed(layer))
            points.extend(layer)
        return points

    raise ValueError("unknown trajectory: {}".format(args.trajectory))
