#!/usr/bin/env python3
"""Grid-distributed ORB map detector adapted from orb.py."""

from __future__ import annotations

import cv2
import numpy as np


MAP_FILE = "map.jpg"


class OrbDetector:
    def __init__(
        self,
        map_file: str = MAP_FILE,
        map_size_m: tuple[float, float] = (3.0, 3.0),
        show_keypoints: bool = False,
    ) -> None:
        self.map_im = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
        if self.map_im is None:
            raise FileNotFoundError("Cannot read ORB map image: {}".format(map_file))

        self.map_h, self.map_w = self.map_im.shape[:2]
        self.map_size_m = map_size_m

        self.orb = cv2.ORB_create(
            nfeatures=200,
            scaleFactor=1.2,
            nlevels=12,
            patchSize=31,
            edgeThreshold=31,
            scoreType=cv2.ORB_HARRIS_SCORE,
            fastThreshold=20,
        )

        self.frame_orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=12,
            patchSize=31,
            edgeThreshold=31,
            scoreType=cv2.ORB_HARRIS_SCORE,
            fastThreshold=15,
        )

        self.map_kp, self.map_des = self._detect_orb_grid(5, 5)
        if self.map_des is None or not self.map_kp:
            raise RuntimeError("ORB map has no descriptors: {}".format(map_file))

        self.last_homography: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_frame_kp = []
        self.last_good_matches = 0
        self.last_inliers = 0

        if show_keypoints:
            debug = cv2.drawKeypoints(
                self.map_im,
                self.map_kp,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            )
            cv2.imshow("ORB Keypoints", cv2.resize(debug, None, fx=0.4, fy=0.4))
            cv2.waitKey(0)

    def _detect_orb_grid(self, grid_rows, grid_cols):
        h, w = self.map_im.shape[:2]
        all_keypoints = []
        all_descriptors = []

        row_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, w, grid_cols + 1, dtype=int)

        for r in range(grid_rows):
            for c in range(grid_cols):
                y_min, y_max = row_edges[r], row_edges[r + 1]
                x_min, x_max = col_edges[c], col_edges[c + 1]

                mask = np.zeros(self.map_im.shape, dtype=np.uint8)
                mask[y_min:y_max, x_min:x_max] = 255

                kp, des = self.orb.detectAndCompute(self.map_im, mask)
                if kp:
                    all_keypoints.extend(kp)
                if des is not None and len(des) > 0:
                    all_descriptors.append(des)
        descriptors_stacked = np.vstack(all_descriptors) if all_descriptors else None
        return all_keypoints, descriptors_stacked

    def _estimate_homography(self, frame_gray):
        frame_kp, frame_des = self.frame_orb.detectAndCompute(frame_gray, None)

        self.last_homography = None
        self.last_mask = None
        self.last_frame_kp = frame_kp or []
        self.last_good_matches = 0
        self.last_inliers = 0

        if frame_des is None or self.map_des is None:
            return None, None, frame_kp

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(frame_des, self.map_des, k=2)

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        self.last_good_matches = len(good)

        if len(good) < 8:
            return None, None, frame_kp

        src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.map_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        self.last_homography = h
        self.last_mask = mask
        if mask is not None:
            self.last_inliers = int(mask.ravel().sum())
        return h, mask, frame_kp

    def _map_px_to_m(self, pt_px):
        x_m = pt_px[0] / self.map_w * self.map_size_m[0]
        y_m = pt_px[1] / self.map_h * self.map_size_m[1]
        return np.array([x_m, y_m], dtype=np.float32)

    def _coordinates_from_homography(self, frame_gray, H, point=None, in_meters=False):
        h, w = frame_gray.shape[:2]
        if point is None:
            point = (w / 2.0, h / 2.0)

        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        mapped_px = cv2.perspectiveTransform(pts, H)[0, 0]

        p1 = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
        p2 = np.array([[[w / 2.0 + 50.0, h / 2.0]]], dtype=np.float32)

        q1 = cv2.perspectiveTransform(p1, H)[0, 0]
        q2 = cv2.perspectiveTransform(p2, H)[0, 0]

        dx = q2[0] - q1[0]
        dy = q2[1] - q1[1]
        angle = np.arctan2(dy, dx)

        if in_meters:
            mapped_m = self._map_px_to_m(mapped_px)
            return mapped_m, angle

        return mapped_px, angle

    def get_frame_coordinates(self, frame, point=None, in_meters=False):
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        H, mask, frame_kp = self._estimate_homography(frame_gray)
        if H is None:
            return None

        return self._coordinates_from_homography(frame_gray, H, point=point, in_meters=in_meters)

    def draw_debug(self, frame):
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()

        H, mask, frame_kp = self._estimate_homography(frame_gray)
        vis = frame.copy()

        if H is not None:
            h, w = frame_gray.shape[:2]
            center = (w / 2.0, h / 2.0)
            mapped_px, mapped_angle_px = self._coordinates_from_homography(
                frame_gray,
                H,
                point=center,
                in_meters=False,
            )
            mapped_m, mapped_angle_m = self._coordinates_from_homography(
                frame_gray,
                H,
                point=center,
                in_meters=True,
            )

            cv2.drawKeypoints(
                vis,
                frame_kp,
                vis,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            )
            cv2.putText(
                vis,
                f"px: {mapped_px[0]:.1f}, {mapped_px[1]:.1f}, {mapped_angle_px:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"m: {mapped_m[0]:.2f}, {mapped_m[1]:.2f}, {mapped_angle_m:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        return vis
