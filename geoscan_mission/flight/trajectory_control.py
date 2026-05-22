#!/usr/bin/env python3
"""Manual-speed trajectory control for Pioneer-SDK2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol, Sequence


Point3 = tuple[float, float, float]


class RecorderLike(Protocol):
    def raise_if_failed(self) -> None:
        ...


@dataclass
class PIDController:
    kp: float = 1.0
    ki: float = 0.02
    kd: float = 0.1
    output_limit: float = 1.5

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_call = True

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_call = True

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 0.05

        p_term = self.kp * error

        self.integral += error * dt
        integral_limit = self.output_limit / self.ki if self.ki > 0 else self.output_limit
        self.integral = clip(self.integral, -integral_limit, integral_limit)
        i_term = self.ki * self.integral

        if self.first_call:
            d_term = 0.0
            self.first_call = False
        else:
            d_term = self.kd * ((error - self.previous_error) / dt)

        self.previous_error = error
        return clip(p_term + i_term + d_term, -self.output_limit, self.output_limit)


@dataclass
class ManualSpeedControllerConfig:
    max_xy_speed: float = 0.15
    max_z_speed: float = 0.15
    max_yaw_rate: float = 0.7
    position_tolerance: float = 0.1
    control_interval: float = 0.05
    command_interval: float = 0.2
    kp_xy: float = 1.2
    ki_xy: float = 0.03
    kd_xy: float = 0.15
    kp_z: float = 1.0
    ki_z: float = 0.02
    kd_z: float = 0.1
    kp_yaw: float = 1.6
    ki_yaw: float = 0.0
    kd_yaw: float = 0.15
    kp_lateral: float = 1.2
    trajectory_finish_tolerance: float = 0.12


@dataclass(frozen=True)
class TrajectoryFollowCommand:
    vx_body: float
    vy_body: float
    vz: float
    yaw_rate: float
    target_index: int
    target_point: Point3
    yaw_target: float
    yaw_feed_forward: float
    yaw_error: float
    lateral_error: float
    complete: bool


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def transform_global_to_body_fixed(vx_global: float, vy_global: float, yaw_rad: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    vx_body = vx_global * cos_yaw + vy_global * sin_yaw
    vy_body = -vx_global * sin_yaw + vy_global * cos_yaw
    return vx_body, vy_body


def limit_xy_speed(vx: float, vy: float, max_speed: float) -> tuple[float, float]:
    norm = math.hypot(vx, vy)
    if norm <= max_speed or norm == 0:
        return vx, vy
    scale = max_speed / norm
    return vx * scale, vy * scale


def distance_3d(a: Point3, b: Point3) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def horizontal_distance(a: Point3, b: Point3) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_length(points: Sequence[Point3]) -> float:
    return sum(distance_3d(points[index - 1], points[index]) for index in range(1, len(points)))


def nearest_future_index(points: Sequence[Point3], position: Point3, start_index: int) -> int:
    if not points:
        raise ValueError("trajectory must contain at least one point")

    start_index = min(max(start_index, 0), len(points) - 1)
    best_index = start_index
    best_distance = float("inf")
    px, py, pz = position
    for index in range(start_index, len(points)):
        point = points[index]
        distance = (point[0] - px) ** 2 + (point[1] - py) ** 2 + (point[2] - pz) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def tangent_yaw(points: Sequence[Point3], index: int) -> float:
    if len(points) < 2:
        return 0.0

    index = min(max(index, 0), len(points) - 1)
    if index == 0:
        before = points[0]
        after = points[1]
    elif index >= len(points) - 1:
        before = points[-2]
        after = points[-1]
    else:
        before = points[index - 1]
        after = points[index + 1]

    dx = after[0] - before[0]
    dy = after[1] - before[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def yaw_feed_forward(points: Sequence[Point3], index: int, speed: float) -> float:
    if len(points) < 3 or speed <= 0:
        return 0.0

    previous_index = max(0, index - 1)
    next_index = min(len(points) - 1, index + 1)
    if previous_index == next_index:
        return 0.0

    previous_yaw = tangent_yaw(points, previous_index)
    next_yaw = tangent_yaw(points, next_index)
    ds = horizontal_distance(points[previous_index], points[next_index])
    if ds <= 1e-6:
        return 0.0
    return normalize_angle_rad(next_yaw - previous_yaw) * speed / ds


class ManualSpeedTrajectoryController:
    """Closed-loop waypoint controller using SDK2 manual speed commands."""

    def __init__(self, drone, config: ManualSpeedControllerConfig | None = None) -> None:
        self.drone = drone
        self.config = config or ManualSpeedControllerConfig()
        self.validate_drone_api()
        self.pid_x = PIDController(self.config.kp_xy, self.config.ki_xy, self.config.kd_xy, self.config.max_xy_speed)
        self.pid_y = PIDController(self.config.kp_xy, self.config.ki_xy, self.config.kd_xy, self.config.max_xy_speed)
        self.pid_z = PIDController(self.config.kp_z, self.config.ki_z, self.config.kd_z, self.config.max_z_speed)
        self.pid_yaw = PIDController(
            self.config.kp_yaw,
            self.config.ki_yaw,
            self.config.kd_yaw,
            self.config.max_yaw_rate,
        )

    def validate_drone_api(self) -> None:
        required = ["get_local_position_lps", "get_local_yaw_lps", "set_manual_speed_body_fixed"]
        missing = [name for name in required if not hasattr(self.drone, name)]
        if missing:
            raise RuntimeError("manual-speed mode requires Pioneer-SDK2 methods: {}".format(", ".join(missing)))

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

    def get_position(self) -> tuple[float, float, float]:
        if not hasattr(self.drone, "get_local_position_lps"):
            raise RuntimeError("manual-speed mode requires pioneer.get_local_position_lps()")

        position = self.drone.get_local_position_lps()
        if position is None or len(position) < 3:
            raise RuntimeError("get_local_position_lps() returned no valid position")

        x, y, z = float(position[0]), float(position[1]), float(position[2])
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise RuntimeError("get_local_position_lps() returned non-finite values: {}".format(position))
        return x, y, z

    def get_yaw_rad(self) -> float:
        if not hasattr(self.drone, "get_local_yaw_lps"):
            raise RuntimeError("manual-speed mode requires pioneer.get_local_yaw_lps()")

        yaw_deg = self.drone.get_local_yaw_lps()
        if yaw_deg is None:
            raise RuntimeError("get_local_yaw_lps() returned no valid yaw")
        yaw_deg = float(yaw_deg)
        if not math.isfinite(yaw_deg):
            raise RuntimeError("get_local_yaw_lps() returned non-finite yaw: {}".format(yaw_deg))
        return math.radians(yaw_deg)

    def send_manual_speed(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        if not hasattr(self.drone, "set_manual_speed_body_fixed"):
            raise RuntimeError("manual-speed mode requires pioneer.set_manual_speed_body_fixed()")

        try:
            result = self.drone.set_manual_speed_body_fixed(
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_rate=yaw_rate,
                interval=self.config.command_interval,
            )
        except TypeError:
            result = self.drone.set_manual_speed_body_fixed(vx, vy, vz, yaw_rate)

        if result is False:
            raise RuntimeError("set_manual_speed_body_fixed() returned False")

    def hold_stop(self) -> None:
        self.send_manual_speed(0.0, 0.0, 0.0, 0.0)

    def step_towards(self, target: tuple[float, float, float], dt: float) -> bool:
        current_x, current_y, current_z = self.get_position()
        current_yaw = self.get_yaw_rad()
        target_x, target_y, target_z = target

        error_x = target_x - current_x
        error_y = target_y - current_y
        error_z = target_z - current_z
        horizontal_distance = math.hypot(error_x, error_y)
        full_distance = math.sqrt(horizontal_distance * horizontal_distance + error_z * error_z)

        if full_distance <= self.config.position_tolerance:
            self.hold_stop()
            self.reset()
            return True

        desired_yaw = math.atan2(error_y, error_x) if horizontal_distance > 0.01 else current_yaw
        yaw_error = normalize_angle_rad(desired_yaw - current_yaw)

        vx_global = self.pid_x.compute(error_x, dt)
        vy_global = self.pid_y.compute(error_y, dt)
        vx_global, vy_global = limit_xy_speed(vx_global, vy_global, self.config.max_xy_speed)
        vz_global = clip(self.pid_z.compute(error_z, dt), -self.config.max_z_speed, self.config.max_z_speed)
        yaw_rate = self.pid_yaw.compute(yaw_error, dt)

        vx_body, vy_body = transform_global_to_body_fixed(vx_global, vy_global, current_yaw)
        self.send_manual_speed(vx_body, vy_body, vz_global, yaw_rate)
        return False

    def compute_spline_command(
        self,
        points: Sequence[Point3],
        current_position: Point3,
        current_yaw: float,
        speed: float,
        progress_index: int,
    ) -> TrajectoryFollowCommand:
        if len(points) < 2:
            raise ValueError("spline trajectory requires at least two points")
        if speed <= 0:
            raise ValueError("speed must be positive")

        target_index = nearest_future_index(points, current_position, progress_index)
        target_point = points[target_index]
        final_point = points[-1]
        final_distance = distance_3d(current_position, final_point)
        complete = target_index >= len(points) - 2 and final_distance <= self.config.trajectory_finish_tolerance

        yaw_target = tangent_yaw(points, target_index)
        to_path_x = target_point[0] - current_position[0]
        to_path_y = target_point[1] - current_position[1]
        lateral_error = -to_path_x * math.sin(yaw_target) + to_path_y * math.cos(yaw_target)
        z_error = target_point[2] - current_position[2]
        yaw_error = normalize_angle_rad(yaw_target - current_yaw)
        feed_forward = yaw_feed_forward(points, target_index, speed)

        vx_body = clip(speed, 0.0, self.config.max_xy_speed)
        vy_body = clip(
            self.config.kp_lateral * lateral_error,
            -self.config.max_xy_speed,
            self.config.max_xy_speed,
        )
        vz = clip(self.config.kp_z * z_error, -self.config.max_z_speed, self.config.max_z_speed)
        yaw_rate = clip(
            feed_forward + self.config.kp_yaw * yaw_error,
            -self.config.max_yaw_rate,
            self.config.max_yaw_rate,
        )

        if complete:
            vx_body = 0.0
            vy_body = 0.0
            vz = 0.0
            yaw_rate = 0.0

        return TrajectoryFollowCommand(
            vx_body=vx_body,
            vy_body=vy_body,
            vz=vz,
            yaw_rate=yaw_rate,
            target_index=target_index,
            target_point=target_point,
            yaw_target=yaw_target,
            yaw_feed_forward=feed_forward,
            yaw_error=yaw_error,
            lateral_error=lateral_error,
            complete=complete,
        )

    def follow_spline_trajectory(
        self,
        points: Sequence[Point3],
        speed: float,
        stop_event: threading.Event,
        recorder: RecorderLike | None = None,
        timeout: float | None = None,
    ) -> bool:
        if len(points) < 2:
            raise ValueError("spline trajectory requires at least two points")
        if speed <= 0:
            raise ValueError("speed must be positive")

        self.reset()
        start = time.monotonic()
        progress_index = 0

        try:
            while not stop_event.is_set():
                if timeout is not None and time.monotonic() - start >= timeout:
                    return False
                if recorder is not None:
                    recorder.raise_if_failed()

                current_position = self.get_position()
                current_yaw = self.get_yaw_rad()
                command = self.compute_spline_command(
                    points=points,
                    current_position=current_position,
                    current_yaw=current_yaw,
                    speed=speed,
                    progress_index=progress_index,
                )
                progress_index = max(progress_index, command.target_index)

                if command.complete:
                    self.hold_stop()
                    return True

                self.send_manual_speed(command.vx_body, command.vy_body, command.vz, command.yaw_rate)
                time.sleep(self.config.control_interval)
            return False
        finally:
            self.hold_stop()

    def fly_to_point(
        self,
        target: tuple[float, float, float],
        timeout: float,
        stop_event: threading.Event,
        recorder: RecorderLike | None = None,
    ) -> bool:
        self.reset()
        start = time.monotonic()
        last = start

        try:
            while not stop_event.is_set() and time.monotonic() - start < timeout:
                if recorder is not None:
                    recorder.raise_if_failed()

                now = time.monotonic()
                reached = self.step_towards(target, now - last)
                last = now
                if reached:
                    return True

                time.sleep(self.config.control_interval)
            return False
        finally:
            self.hold_stop()


DroneController = ManualSpeedTrajectoryController


__all__ = [
    "DroneController",
    "ManualSpeedControllerConfig",
    "ManualSpeedTrajectoryController",
    "PIDController",
    "TrajectoryFollowCommand",
    "horizontal_distance",
    "limit_xy_speed",
    "normalize_angle_rad",
    "path_length",
    "tangent_yaw",
    "transform_global_to_body_fixed",
    "yaw_feed_forward",
]
