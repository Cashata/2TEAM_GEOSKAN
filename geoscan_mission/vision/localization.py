#!/usr/bin/env python3
"""ORB/RANSAC map localization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from geoscan_mission.vision.orb_grid import OrbDetector


@dataclass
class LocalizeResult:
    ok: bool
    message: str
    x_m: float | None = None
    y_m: float | None = None
    raw_x_m: float | None = None
    raw_y_m: float | None = None
    smooth_x_m: float | None = None
    smooth_y_m: float | None = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    angle_rad: float | None = None
    homography_area: float | None = None
    accepted_by_filter: bool = False
    filter_reason: str = ""


class OrbRansacLocalizer:
    def __init__(
        self,
        reference_path: str,
        map_width_m: float,
        map_height_m: float,
        feature: str,
        nfeatures: int,
        ratio: float,
        min_matches: int,
        min_inliers: int,
        ransac_threshold: float,
        frame_width: int,
        frame_height: int,
        reference_max_size: int,
        clahe_clip: float,
        clahe_tile: int,
        min_inlier_ratio: float,
        min_homography_area_m2: float,
        max_homography_area_m2: float,
        max_position_jump: float,
        ema_alpha: float,
    ) -> None:
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.feature = feature.lower()
        if self.feature != "orb":
            raise ValueError("Only --feature orb is supported in the flight pipeline")
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.frame_size = (frame_width, frame_height)
        self.min_inlier_ratio = min_inlier_ratio
        self.min_homography_area_m2 = min_homography_area_m2
        self.max_homography_area_m2 = (
            max_homography_area_m2 if max_homography_area_m2 > 0 else map_width_m * map_height_m * 1.1
        )
        self.max_position_jump = max_position_jump
        self.ema_alpha = ema_alpha
        self.smooth_x_m: float | None = None
        self.smooth_y_m: float | None = None
        self.first_valid_for_waypoint = True
        self.orb_grid: OrbDetector | None = None
        self.orb_grid = OrbDetector(
            map_file=reference_path,
            map_size_m=(map_width_m, map_height_m),
        )
        self.reference = self.orb_grid.map_im
        self.ref_h, self.ref_w = self.reference.shape[:2]

    @staticmethod
    def resize_reference(reference: np.ndarray, max_size: int) -> np.ndarray:
        if max_size <= 0:
            return reference
        height, width = reference.shape[:2]
        longest = max(width, height)
        if longest <= max_size:
            return reference
        scale = max_size / float(longest)
        return cv2.resize(reference, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)

    def prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        width, height = self.frame_size
        if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
            return frame_bgr
        return cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_AREA)

    def start_waypoint(self) -> None:
        self.first_valid_for_waypoint = True

    def ref_pixel_to_map_m(self, px: float, py: float) -> tuple[float, float]:
        x_m = px / self.ref_w * self.map_width_m
        y_m = py / self.ref_h * self.map_height_m
        return float(x_m), float(y_m)

    def homography_area_m2(self, homography: np.ndarray, frame_shape: tuple[int, int]) -> float | None:
        height, width = frame_shape
        corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
        projected = cv2.perspectiveTransform(corners, homography)[0]
        if not np.all(np.isfinite(projected)):
            return None
        area_px = abs(cv2.contourArea(projected.astype(np.float32)))
        pixel_area_m2 = (self.map_width_m / self.ref_w) * (self.map_height_m / self.ref_h)
        return float(area_px * pixel_area_m2)

    def last_smooth(self) -> tuple[float | None, float | None]:
        return self.smooth_x_m, self.smooth_y_m

    def reject_result(
        self,
        message: str,
        good_matches: int = 0,
        inliers: int = 0,
        inlier_ratio: float = 0.0,
        homography_area: float | None = None,
        raw_x_m: float | None = None,
        raw_y_m: float | None = None,
    ) -> LocalizeResult:
        return LocalizeResult(
            False,
            message,
            raw_x_m=raw_x_m,
            raw_y_m=raw_y_m,
            smooth_x_m=self.smooth_x_m,
            smooth_y_m=self.smooth_y_m,
            good_matches=good_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            homography_area=homography_area,
            accepted_by_filter=False,
            filter_reason=message,
        )

    def accept_result(
        self,
        raw_x_m: float,
        raw_y_m: float,
        good_matches: int,
        inliers: int,
        inlier_ratio: float,
        homography_area: float,
    ) -> LocalizeResult:
        if self.smooth_x_m is None or self.smooth_y_m is None:
            smooth_x_m = raw_x_m
            smooth_y_m = raw_y_m
        else:
            smooth_x_m = (1.0 - self.ema_alpha) * self.smooth_x_m + self.ema_alpha * raw_x_m
            smooth_y_m = (1.0 - self.ema_alpha) * self.smooth_y_m + self.ema_alpha * raw_y_m

        self.smooth_x_m = smooth_x_m
        self.smooth_y_m = smooth_y_m
        self.first_valid_for_waypoint = False

        return LocalizeResult(
            True,
            "ok",
            x_m=smooth_x_m,
            y_m=smooth_y_m,
            raw_x_m=raw_x_m,
            raw_y_m=raw_y_m,
            smooth_x_m=smooth_x_m,
            smooth_y_m=smooth_y_m,
            good_matches=good_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            angle_rad=None,
            homography_area=homography_area,
            accepted_by_filter=True,
            filter_reason="ok",
        )

    def estimate(self, frame_bgr: np.ndarray) -> tuple[LocalizeResult, np.ndarray | None, np.ndarray]:
        frame_bgr = self.prepare_frame(frame_bgr)
        return self.estimate_orb_grid(frame_bgr)

    def estimate_orb_grid(self, frame_bgr: np.ndarray) -> tuple[LocalizeResult, np.ndarray | None, np.ndarray]:
        assert self.orb_grid is not None

        if len(frame_bgr.shape) == 3:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_bgr

        homography, mask, _ = self.orb_grid._estimate_homography(frame_gray)
        good_matches = self.orb_grid.last_good_matches
        inliers = self.orb_grid.last_inliers
        inlier_ratio = inliers / max(good_matches, 1)

        if homography is None or mask is None:
            return self.reject_result(
                "ORB grid homography failed",
                good_matches=good_matches,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
            ), None, frame_bgr

        homography_area = self.homography_area_m2(homography, frame_gray.shape[:2])
        coordinates = self.orb_grid._coordinates_from_homography(frame_gray, homography, in_meters=True)
        if coordinates is None:
            return (
                self.reject_result(
                    "ORB grid coordinate projection failed",
                    good_matches=good_matches,
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                ),
                homography,
                frame_bgr,
            )

        mapped_m, angle_rad = coordinates
        raw_x_m, raw_y_m = float(mapped_m[0]), float(mapped_m[1])
        if not (0.0 <= raw_x_m <= self.map_width_m and 0.0 <= raw_y_m <= self.map_height_m):
            return (
                self.reject_result(
                    "estimated center is outside reference map",
                    good_matches=good_matches,
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                    raw_x_m=raw_x_m,
                    raw_y_m=raw_y_m,
                ),
                homography,
                frame_bgr,
            )

        if (
            self.max_position_jump > 0
            and not self.first_valid_for_waypoint
            and self.smooth_x_m is not None
            and self.smooth_y_m is not None
            and math.dist((raw_x_m, raw_y_m), (self.smooth_x_m, self.smooth_y_m)) > self.max_position_jump
        ):
            return (
                self.reject_result(
                    "position jump too large",
                    good_matches=good_matches,
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    homography_area=homography_area,
                    raw_x_m=raw_x_m,
                    raw_y_m=raw_y_m,
                ),
                homography,
                frame_bgr,
            )

        result = self.accept_result(
            raw_x_m,
            raw_y_m,
            good_matches,
            inliers,
            inlier_ratio,
            homography_area,
        )
        result.angle_rad = float(angle_rad)
        result.filter_reason = "ok"
        return result, homography, frame_bgr

    def draw_debug(self, frame_bgr: np.ndarray, result: LocalizeResult, homography: np.ndarray | None) -> np.ndarray:
        ref_debug = cv2.cvtColor(self.reference, cv2.COLOR_GRAY2BGR)

        if homography is not None:
            h, w = frame_bgr.shape[:2]
            corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
            projected = cv2.perspectiveTransform(corners, homography)
            if np.all(np.isfinite(projected)):
                cv2.polylines(ref_debug, [np.int32(projected[0])], True, (255, 0, 0), 2)

        point_x = result.x_m if result.ok else result.raw_x_m
        point_y = result.y_m if result.ok else result.raw_y_m
        if point_x is not None and point_y is not None:
            px = int(round(point_x / self.map_width_m * self.ref_w))
            py = int(round(point_y / self.map_height_m * self.ref_h))
            color = (0, 255, 0) if result.ok else (0, 0, 255)
            cv2.circle(ref_debug, (px, py), 8, color, -1)

        if result.ok and result.x_m is not None and result.y_m is not None:
            text = "x={:.2f}m y={:.2f}m inliers={}".format(result.x_m, result.y_m, result.inliers)
        else:
            text = "{} matches={} inliers={}".format(result.message, result.good_matches, result.inliers)

        cv2.putText(ref_debug, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return ref_debug
