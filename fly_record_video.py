#!/usr/bin/env python3
"""Ready flight profile: small spline square with video recording enabled."""

from __future__ import annotations

from datetime import datetime
import sys

from geoscan_mission.cli.fly_orb_ransac import main


def build_default_args() -> list[str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return [
        "--reference",
        "map.jpg",
        "--camera-source",
        "sdk2",
        "--sdk2-camera-type",
        "OPT",
        "--control-mode",
        "manual-speed",
        "--trajectory",
        "square",
        "--trajectory-points",
        "5000",
        "--area-size",
        "0.6",
        "--margin",
        "0.2",
        "--height",
        "0.6",
        "--speed",
        "0.08",
        "--aruco",
        "--csv",
        f"flights/{stamp}_localization.csv",
        "--events-log",
        f"flights/{stamp}_events.csv",
        "--video-camera-out",
        f"flights/{stamp}_camera_overlay.avi",
        "--video-map-out",
        f"flights/{stamp}_map_trace.avi",
    ]


if __name__ == "__main__":
    raise SystemExit(main(build_default_args() + sys.argv[1:]))
