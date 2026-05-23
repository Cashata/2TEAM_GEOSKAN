import time

import cv2
from pioneer_sdk2 import Pioneer, Camera, CameraType

from orb import OrbDetector

ind = 0


def save():
    global ind
    ind += 1

    print("TAKE PHOTO")
    frame = camera.get_cv_frame(timeout=5.0)

    if frame is None:
        print("ERROR: camera frame is empty")
    else:
        cv2.imwrite(f"snapshot_{ind}.png", frame)
        cv2.imwrite(f"snapshot_{ind}orb.png", orb.draw_debug(frame))
        print(f"Saved: snapshot_{ind}.png")


drone = Pioneer()
camera = Camera(camera_type=CameraType.OPT)

print("orb create")
orb = OrbDetector()

try:
    print("ARM")
    drone.arm()
    print("TAKEOFF")
    drone.takeoff()
    time.sleep(10)

    save()

    print("GO TO 1 METER")
    drone.go_to_local_point(x=0.0, y=0.0, z=1.0, yaw=0.0, time=3)
    time.sleep(10)
    save()

    print("GO TO 1.5 METER")
    drone.go_to_local_point(x=0.0, y=0.0, z=1.5, yaw=0.0, time=3)
    time.sleep(10)
    save()

    print("GO TO 2 METER")
    drone.go_to_local_point(x=0.0, y=0.0, z=2, yaw=0.0, time=3)
    time.sleep(10)
    save()

    time.sleep(1)

    print("LAND")
    drone.land()
    drone.disarm()
finally:
    camera.stop()
