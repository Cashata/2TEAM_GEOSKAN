#!/usr/bin/env python3
"""Compatibility re-export for the modular ArUco detector."""

from geoscan_mission.vision.aruco import (
    DEFAULT_DICTIONARY,
    DEFAULT_TARGETS,
    ArucoDetector,
    ArucoMarker,
    create_detector_parameters,
    resolve_aruco_dictionary,
)

__all__ = [
    "DEFAULT_DICTIONARY",
    "DEFAULT_TARGETS",
    "ArucoDetector",
    "ArucoMarker",
    "create_detector_parameters",
    "resolve_aruco_dictionary",
]
