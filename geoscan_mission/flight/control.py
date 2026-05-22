#!/usr/bin/env python3
"""Flight control helpers around Pioneer-SDK2."""

from __future__ import annotations

import importlib
import math
import sys
import threading
import time


RTL_COMMANDS = {
    "rtl",
    "rth",
    "return",
    "return_home",
    "return-to-home",
    "return_to_home",
    "return-to-launch",
    "return_to_launch",
}

LAND_COMMANDS = {
    "land",
    "stop",
    "q",
    "quit",
    "exit",
    "posadka",
    "sest",
    "посадка",
    "сесть",
    "стоп",
}


class FlightCommandState:
    """Thread-safe storage for the last requested console flight action."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action: str | None = None
        self._command: str | None = None

    def request(self, action: str, command: str) -> None:
        with self._lock:
            self._action = action
            self._command = command

    def get_action(self) -> str | None:
        with self._lock:
            return self._action

    def get_command(self) -> str | None:
        with self._lock:
            return self._command


def import_pioneer_sdk2():
    try:
        return importlib.import_module("pioneer_sdk2")
    except ModuleNotFoundError as exc:
        if exc.name != "pioneer_sdk2":
            raise
        raise RuntimeError(
            "pioneer_sdk2 is not installed in this Python environment. "
            "Run real flights on Pioneer Mini 2 / Pioneer OS, or use --no-flight locally."
        ) from exc


def create_pioneer(sdk2):
    try:
        return sdk2.Pioneer(wait_callback=True, safety_command=True)
    except TypeError:
        return sdk2.Pioneer()


def warn_show_disabled(show: bool) -> None:
    if show:
        print(
            "WARNING: --show is ignored here because onboard OpenCV may be built without HighGUI. "
            "Use --debug-dir to save debug images instead.",
            file=sys.stderr,
        )


def start_command_listener(
    stop_event: threading.Event,
    command_state: FlightCommandState | None = None,
) -> threading.Thread:
    def listen() -> None:
        print("Type 'rtl', 'land', 'stop', 'q', 'posadka' or 'sest' + Enter to stop the route safely.")
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except OSError:
                return
            if line == "":
                return

            command = line.strip().lower()
            if not command:
                continue
            if command in RTL_COMMANDS:
                if command_state is not None:
                    command_state.request("rtl", command)
                print("Return-to-launch requested by command: {}".format(command))
                stop_event.set()
                return
            if command in LAND_COMMANDS:
                if command_state is not None:
                    command_state.request("land", command)
                print("Graceful landing requested by command: {}".format(command))
                stop_event.set()
                return
            commands = sorted(RTL_COMMANDS | LAND_COMMANDS)
            print("Unknown command '{}'. Use: {}".format(command, ", ".join(commands)))

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return thread


def estimate_move_time(
    previous_point: tuple[float, float, float],
    next_point: tuple[float, float, float],
    speed: float,
) -> int:
    distance = math.dist(previous_point, next_point)
    return max(1, int(math.ceil(distance / speed)))


def command_local_point(drone, x: float, y: float, z: float, yaw: float, point_time: int) -> None:
    try:
        result = drone.go_to_local_point(x=x, y=y, z=z, yaw=yaw, time=point_time)
    except TypeError:
        result = drone.go_to_local_point(x, y, z, yaw, point_time)

    if result is False:
        print("WARNING: go_to_local_point returned False", file=sys.stderr)


def parse_battery_status(status) -> tuple[float, float] | None:
    if status is None:
        return None

    if isinstance(status, dict):
        voltage = status.get("voltage", status.get("battery_voltage"))
        temperature = status.get("temperature", status.get("battery_temperature"))
    elif isinstance(status, (list, tuple)) and len(status) >= 2:
        voltage, temperature = status[0], status[1]
    else:
        return None

    try:
        return float(voltage), float(temperature)
    except (TypeError, ValueError):
        return None


def check_battery_or_abort(
    drone,
    min_voltage: float,
    retries: int,
    retry_delay: float,
) -> None:
    if min_voltage <= 0:
        print("Battery voltage check disabled")
        return

    if not hasattr(drone, "get_battery_status"):
        print("WARNING: get_battery_status() unavailable")
        return

    parsed_status = None
    last_status = None
    for attempt in range(1, retries + 1):
        last_status = drone.get_battery_status()
        print("Battery status attempt {}/{}: {}".format(attempt, retries, last_status))
        parsed_status = parse_battery_status(last_status)
        if parsed_status is not None:
            break
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)

    if parsed_status is None:
        raise RuntimeError("Cannot read battery status: {}".format(last_status))

    voltage, temperature = parsed_status
    print("Battery voltage={:.2f} V, temperature={:.1f} C".format(voltage, temperature))

    if voltage < min_voltage:
        raise RuntimeError(
            "Battery voltage too low for flight: {:.2f} V < {:.2f} V".format(voltage, min_voltage)
        )


def wait_for_point(
    drone,
    timeout: float,
    poll_interval: float,
    stop_event: threading.Event,
) -> bool:
    if not hasattr(drone, "point_reached"):
        time.sleep(timeout)
        return True

    start = time.monotonic()
    while time.monotonic() - start < timeout and not stop_event.is_set():
        try:
            if drone.point_reached():
                return True
        except Exception as exc:
            print("WARNING: point_reached() failed: {}".format(exc), file=sys.stderr)
            break
        time.sleep(poll_interval)

    return False


def call_builtin_rtl(drone) -> bool:
    for method_name in ("rtl", "return_to_launch", "return_to_home"):
        method = getattr(drone, method_name, None)
        if not callable(method):
            continue

        result = method()
        if result is False:
            raise RuntimeError("{}() returned False".format(method_name))
        print("Sent return-to-launch command via {}().".format(method_name))
        return True
    return False


def return_to_launch_or_land(
    drone,
    height: float,
    yaw: float,
    point_time: int,
    timeout: float,
    poll_interval: float,
) -> bool:
    """Run SDK RTL when available, otherwise return to local origin and land."""

    if call_builtin_rtl(drone):
        return True

    print("WARNING: no rtl() method found; falling back to x=0 y=0 return and landing.", file=sys.stderr)
    if hasattr(drone, "go_to_local_point"):
        command_local_point(drone, 0.0, 0.0, max(height, 0.1), yaw=yaw, point_time=point_time)
        wait_for_point(
            drone=drone,
            timeout=timeout,
            poll_interval=poll_interval,
            stop_event=threading.Event(),
        )

    if hasattr(drone, "land"):
        drone.land()
    return False


def sleep_while_recording(seconds: float, recorder: ContinuousFlightRecorder, stop_event: threading.Event) -> None:
    start = time.monotonic()
    while time.monotonic() - start < seconds and not stop_event.is_set():
        recorder.raise_if_failed()
        remaining = seconds - (time.monotonic() - start)
        time.sleep(min(0.1, max(remaining, 0.0)))
