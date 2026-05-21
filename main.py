import time

import cv2
import numpy as np

from PathFinder import PathFinder
from SmoothPath import SmoothPath

MAP_FILE = 'map-2.png'
MAP_SCALE_FACTOR = 0.25
DEAD_ZONE_COORD = np.array([1015, 1230]) * MAP_SCALE_FACTOR
DEAD_ZONE_RADIUS = 550 * MAP_SCALE_FACTOR

map_im = cv2.resize(cv2.imread(MAP_FILE, cv2.IMREAD_GRAYSCALE), None,
                    fx=MAP_SCALE_FACTOR, fy=MAP_SCALE_FACTOR, interpolation=cv2.INTER_AREA)
map_mask = cv2.circle(np.ones_like(map_im),
                      DEAD_ZONE_COORD.astype(np.uint16), int(DEAD_ZONE_RADIUS), 0, cv2.FILLED)

# cv2.imshow('map', map_im * map_mask)
# cv2.imshow('map', map_mask * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

path_finder = PathFinder(map_mask)

t = time.time()
path = path_finder.find_path((100, 200), (500, 500))
path = SmoothPath(path).path
print(time.time() - t)

print(path)
path_im = cv2.cvtColor(map_mask * 255, cv2.COLOR_GRAY2BGR)
for x, y in path:
    cv2.circle(path_im, (int(x), int(y)), 2, (0, 0, 255), cv2.FILLED)
cv2.imshow('path', path_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
