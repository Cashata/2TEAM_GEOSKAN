#!/usr/bin/env python3
"""Reusable ArUco target detection for the Geoscan mission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_TARGETS: dict[int, dict[str, str]] = {
    3: {"letter": "И", "type": "allowed"},
    23: {"letter": "Т", "type": "allowed"},
    42: {"letter": "М", "type": "forbidden"},
    117: {"letter": "О", "type": "forbidden"},
}

DEFAULT_DICTIONARY = "DICT_4X4_1000"


@dataclass
class ArucoMarker:
    marker_id: int
    letter: str
    target_type: str
    corners_px: list[list[float]]
    center_px: list[float]
    first_seen: bool = False
    center_ref_px: list[float] | None = None
    center_map_m: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.marker_id,
            "letter": self.letter,
            "type": self.target_type,
            "corners_px": self.corners_px,
            "center_px": self.center_px,
            "center_ref_px": self.center_ref_px,
            "center_map_m": self.center_map_m,
            "first_seen": self.first_seen,
        }


def resolve_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without cv2.aruco support")

    name = dictionary_name.strip().upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name

    if not hasattr(cv2.aruco, name):
        available = sorted(item for item in dir(cv2.aruco) if item.startswith("DICT_"))
        raise ValueError(
            "unknown ArUco dictionary '{}'. Available examples: {}".format(
                dictionary_name,
                ", ".join(available[:12]) if available else "none",
            )
        )

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_detector_parameters():
    try:
        return cv2.aruco.DetectorParameters()
    except AttributeError:
        return cv2.aruco.DetectorParameters_create()


class ArucoDetector:
    def __init__(
        self,
        targets: dict[int, dict[str, str]] | None = None,
        dictionary_name: str = DEFAULT_DICTIONARY,
    ) -> None:
        self.targets = targets or DEFAULT_TARGETS
        self.dictionary_name = dictionary_name
        self.aruco_dict = resolve_aruco_dictionary(dictionary_name)
        self.params = create_detector_parameters()
        self.found_markers: dict[int, dict[str, Any]] = {}

        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        else:
            self._detector = None

    def detect_markers(self, frame: np.ndarray):
        if frame is None:
            return [], None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._detector is not None:
            return self._detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

    def process_frame(self, frame: np.ndarray) -> list[ArucoMarker]:
        corners, ids, _ = self.detect_markers(frame)
        if ids is None:
            return []

        detections: list[ArucoMarker] = []
        for index, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self.targets:
                continue

            info = self.targets[marker_id]
            corner = np.asarray(corners[index][0], dtype=np.float64)
            center = corner.mean(axis=0)
            first_seen = marker_id not in self.found_markers
            marker = ArucoMarker(
                marker_id=marker_id,
                letter=info["letter"],
                target_type=info["type"],
                corners_px=corner.tolist(),
                center_px=center.tolist(),
                first_seen=first_seen,
            )
            detections.append(marker)

            if first_seen:
                self.found_markers[marker_id] = marker.as_dict()

        return detections

    def remember_projection(self, marker: ArucoMarker) -> None:
        if marker.marker_id in self.found_markers:
            stored = self.found_markers[marker.marker_id]
            stored["center_ref_px"] = marker.center_ref_px
            stored["center_map_m"] = marker.center_map_m

    def reset(self) -> None:
        self.found_markers.clear()

    def get_word(self) -> str:
        return "".join(self.found_markers[mid]["letter"] for mid in sorted(self.found_markers))

    def found_ids(self, target_type: str | None = None) -> list[int]:
        marker_ids = sorted(self.found_markers)
        if target_type is None:
            return marker_ids
        return [mid for mid in marker_ids if self.found_markers[mid]["type"] == target_type]

    def found_as_list(self) -> list[dict[str, Any]]:
        return [self.found_markers[mid] for mid in sorted(self.found_markers)]
