#!/usr/bin/env python3
"""Manual-speed trajectory control for Pioneer-SDK2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol


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
    "limit_xy_speed",
    "normalize_angle_rad",
    "transform_global_to_body_fixed",
]
