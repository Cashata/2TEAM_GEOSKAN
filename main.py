import os
import threading
import time
from math import dist

import cv2
import numpy as np
from pioneer_sdk2 import Pioneer, Camera, CameraType

from PathFinder import PathFinder
from orb import OrbDetector

drone = Pioneer()
camera = Camera(camera_type=CameraType.OPT)
orb = OrbDetector()

running = True
drone_x = 1000
drone_y = 1000
drone_yaw = 0


def coord_updater():
    global drone_x, drone_y, drone_yaw, running, camera, orb
    os.makedirs("coord_updater", exist_ok=True)
    i = 0

    with open("coord_updater/coord_updater.txt", "w") as f:
        while running:
            frame = camera.get_cv_frame(timeout=1.0)
            ret = orb.get_frame_coordinates(frame)
            if ret is not None:
                coord, yaw = ret
                drone_x = coord[0]
                drone_y = coord[1]
                drone_yaw = yaw

                f.write(f"{time.time()},{drone_x},{drone_y},{drone_yaw}\n")
            else:
                f.write(f"{time.time()},None\n")
                cv2.imwrite(f"coord_updater/none_frame_{i}_{time.time()}.png", frame)
            i += 1
            if i % 2 == 0:
                cv2.imwrite(f"coord_updater/frame_{i}_{time.time()}.png", frame)


def go_path(path):
    global drone, drone_x, drone_y, drone_yaw

    while abs(drone_yaw) > 0.1:
        p = drone_yaw * 2.0
        p = max(-0.5, min(0.5, p))
        print("degree_error", time.time(), drone_yaw, p)
        drone.set_manual_speed_body_fixed(vx=0.0, vy=0.0, vz=0.0, yaw_rate=p, interval=0.1)
        time.sleep(0.1)

    for x, y in path:
        x /= PATH_SCALE
        y /= PATH_SCALE
        with open(f"path_{x}_{y}.txt", "w") as f:
            while dist((drone_x, drone_y), (x, y)) > 40:
                yaw = drone_yaw * 2.0
                yaw = max(-0.5, min(0.5, yaw))

                vx = (drone_x - x) * 0.5
                vx = max(-0.5, min(0.5, vx))

                vy = (y - drone_y) * 0.5
                vy = max(-0.5, min(0.5, vy))

                f.write(f"{time.time()},{vx},{vy},{yaw}\n")
                print("path", time.time(), x, y, vx, vy, yaw)

                drone.set_manual_speed_body_fixed(vx=-vy, vy=-vx, vz=0.0, yaw_rate=yaw, interval=0.1)
                time.sleep(0.1)


MAP_PX = np.array([2500, 2500])
MAP_M = np.array([3, 3])
PATH_SCALE = 0.2
DEAD_ZONE_COORD = np.array([1015, 1230]) * PATH_SCALE
DEAD_ZONE_RADIUS = 400 * PATH_SCALE
map_mask = cv2.circle(np.ones((MAP_PX * PATH_SCALE).astype(np.uint16), np.uint8),
                      DEAD_ZONE_COORD.astype(np.uint16), int(DEAD_ZONE_RADIUS), 0, cv2.FILLED)
path_finder = PathFinder(map_mask)

try:
    print("ARM")
    drone.arm()
    print("TAKEOFF")
    drone.takeoff()
    print("GO TO HIGHT")
    drone.go_to_local_point(x=0.0, y=0.0, z=1, yaw=0.0, time=3)
    time.sleep(10)

    coord_updater_thread = threading.Thread(target=coord_updater)
    coord_updater_thread.start()
    time.sleep(5)

    for x in (0.75, 3 - 0.75):
        for y in (0.75, 3 - 0.75):
            x = int(x / MAP_M[0] * MAP_PX[0])
            y = int(y / MAP_M[1] * MAP_PX[1])

            path = path_finder.find_path((drone_x * PATH_SCALE, drone_y * PATH_SCALE),
                                         (x * PATH_SCALE, y * PATH_SCALE))
            with open(f"path_plan_{x}_{y}.txt", "w") as f:
                for x, y in path:
                    f.write(f"{x},{y}\n")

            go_path(path)
            time.sleep(10)
finally:
    running = False
    print("LAND")
    drone.land()
    drone.disarm()
    camera.stop()
