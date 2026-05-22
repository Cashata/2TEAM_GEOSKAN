#!/usr/bin/env python3
"""Demo wrapper for grid path planning on a map image."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from geoscan_mission.trajectory.grid_path import PathFinder, SmoothPath


DEFAULT_MAP_FILE = "map-2.png"
DEFAULT_MAP_SCALE_FACTOR = 0.25
DEFAULT_DEAD_ZONE_COORD = np.array([1015, 1230]) * DEFAULT_MAP_SCALE_FACTOR
DEFAULT_DEAD_ZONE_RADIUS = 550 * DEFAULT_MAP_SCALE_FACTOR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally draw a smoothed path around a circular dead zone.")
    parser.add_argument("--map", default=DEFAULT_MAP_FILE, help="Input grayscale map image.")
    parser.add_argument("--scale", type=float, default=DEFAULT_MAP_SCALE_FACTOR, help="Map resize scale factor.")
    parser.add_argument("--start", default="100,200", help="Start pixel as x,y after scaling.")
    parser.add_argument("--end", default="500,500", help="End pixel as x,y after scaling.")
    parser.add_argument("--output", help="Optional output image with the path drawn.")
    parser.add_argument("--show", action="store_true", help="Show the path image in an OpenCV window.")
    return parser


def parse_point(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be x,y")
    return parts[0], parts[1]


def build_path(map_file: str, scale: float, start: tuple[int, int], end: tuple[int, int]):
    map_raw = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    if map_raw is None:
        raise FileNotFoundError("Cannot read map image: {}".format(map_file))

    map_im = cv2.resize(map_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dead_zone_coord = np.array([1015, 1230]) * scale
    dead_zone_radius = int(550 * scale)
    map_mask = cv2.circle(np.ones_like(map_im), dead_zone_coord.astype(np.uint16), dead_zone_radius, 0, cv2.FILLED)

    path_finder = PathFinder(map_mask)
    path = path_finder.find_path(start, end)
    return SmoothPath(path).path, map_mask


def draw_path(path, map_mask):
    path_im = cv2.cvtColor(map_mask * 255, cv2.COLOR_GRAY2BGR)
    for x, y in path:
        cv2.circle(path_im, (int(x), int(y)), 2, (0, 0, 255), cv2.FILLED)
    return path_im


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = parse_point(args.start)
    end = parse_point(args.end)

    started = time.time()
    path, map_mask = build_path(args.map, args.scale, start, end)
    print("elapsed={:.3f}s".format(time.time() - started))
    print(path)

    if args.output or args.show:
        path_im = draw_path(path, map_mask)
        if args.output:
            ok = cv2.imwrite(args.output, path_im)
            if not ok:
                raise RuntimeError("Cannot write output image: {}".format(args.output))
        if args.show:
            cv2.imshow("path", path_im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
