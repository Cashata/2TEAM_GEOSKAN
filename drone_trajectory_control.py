#!/usr/bin/env python3
"""Compatibility wrapper for manual-speed trajectory control."""

from __future__ import annotations

import argparse

import numpy as np

from geoscan_mission.flight.trajectory_control import (
    DroneController,
    ManualSpeedControllerConfig,
    ManualSpeedTrajectoryController,
    PIDController,
    transform_global_to_body_fixed,
)
from geoscan_mission.trajectory.grid_path import create_trajectory_from_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo wrapper for geoscan_mission.flight.trajectory_control."
    )
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--start", default="5,5", help="Grid start as x,y.")
    parser.add_argument("--end", default="45,45", help="Grid end as x,y.")
    parser.add_argument("--no-smooth", action="store_true", help="Disable SmoothPath for the printed demo route.")
    return parser


def parse_grid_point(value: str) -> list[int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be x,y")
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = parse_grid_point(args.start)
    end = parse_grid_point(args.end)

    cost_map = np.ones((args.grid_size, args.grid_size), dtype=np.uint16) * 100
    trajectory = create_trajectory_from_grid(cost_map, start, end, smooth=not args.no_smooth)

    print("Manual-speed flight control lives in geoscan_mission.flight.trajectory_control")
    print("Use it through: python fly_orb_ransac.py --control-mode manual-speed ...")
    print("Demo grid trajectory points: {}".format(len(trajectory)))
    print("First 5:", trajectory[:5])
    print("Last 5:", trajectory[-5:])
    return 0


__all__ = [
    "DroneController",
    "ManualSpeedControllerConfig",
    "ManualSpeedTrajectoryController",
    "PIDController",
    "create_trajectory_from_grid",
    "transform_global_to_body_fixed",
]


if __name__ == "__main__":
    raise SystemExit(main())
