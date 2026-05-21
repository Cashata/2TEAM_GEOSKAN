import numpy as np
from scipy.interpolate import splprep, splev


class SmoothPath:
    def __init__(self, path, s=50, k=2, num_points=20):
        tck, u = splprep(path.T, s=s, k=k)
        u_new = np.linspace(0, 1, num_points)
        x, y = splev(u_new, tck)
        self.path = np.column_stack((x, y))
