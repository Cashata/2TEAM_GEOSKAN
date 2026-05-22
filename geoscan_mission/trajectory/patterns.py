#!/usr/bin/env python3
"""Reusable waypoint and trajectory pattern builders."""

from __future__ import annotations

import argparse


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
