#!/usr/bin/env python3
"""Compatibility wrapper for the modular Geoscan mission flight CLI."""

from geoscan_mission.cli.fly_orb_ransac import build_parser, fly_local_waypoints, main, validate_args
from geoscan_mission.flight.camera import OpenCvCamera, Sdk2Camera, resolve_sdk2_camera_type
from geoscan_mission.flight.control import (
    LAND_COMMANDS,
    check_battery_or_abort,
    command_local_point,
    create_pioneer,
    estimate_move_time,
    import_pioneer_sdk2,
    parse_battery_status,
    sleep_while_recording,
    start_command_listener,
    wait_for_point,
    warn_show_disabled,
)
from geoscan_mission.recording import (
    ContinuousFlightRecorder,
    FlightVideoLogger,
    append_csv,
    aruco_frame_summary,
    map_point_to_pixel,
    project_aruco_markers,
    result_row,
    video_path_enabled,
)
from geoscan_mission.trajectory.patterns import (
    DEFAULT_WAYPOINTS,
    build_lawnmower_points,
    build_square_points,
    linspace,
    parse_float_list,
    parse_waypoint,
    resolve_waypoints,
)
from geoscan_mission.vision.aruco import ArucoDetector, ArucoMarker, DEFAULT_DICTIONARY
from geoscan_mission.vision.localization import LocalizeResult, OrbRansacLocalizer


if __name__ == "__main__":
    raise SystemExit(main())
