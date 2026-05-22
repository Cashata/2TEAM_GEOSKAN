#!/usr/bin/env python3
"""Grid path planning helpers used by the trajectory demo."""

from __future__ import annotations

import numpy as np


class PathFinder:
    def __init__(self, cost_map):
        import tcod
        from tcod.path import SimpleGraph

        self._tcod = tcod
        self.cost_map = cost_map.T
        self.graph = SimpleGraph(cost=self.cost_map, cardinal=1000, diagonal=1414, greed=1)

    def find_path(self, start, end):
        from tcod.path import Pathfinder

        pf = Pathfinder(self.graph)
        pf.add_root(start)
        return self._contraction(pf.path_to(end))

    def _contraction(self, path):
        if len(path) <= 2:
            return path

        result = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > current_idx + 1:
                if self._is_clear_line(path[current_idx], path[next_idx]):
                    break
                next_idx -= 1
            result.append(path[next_idx])
            current_idx = next_idx

        return np.array(result, dtype=np.uint16)

    def _is_clear_line(self, start, end):
        for x, y in self._tcod.los.bresenham(start, end):
            if self.cost_map[x, y] <= 0:
                return False
        return True


class SmoothPath:
    def __init__(self, path, s=50, k=2, num_points=20):
        from scipy.interpolate import splprep, splev

        tck, u = splprep(path.T, s=s, k=k)
        u_new = np.linspace(0, 1, num_points)
        x, y = splev(u_new, tck)
        self.path = np.column_stack((x, y))


def create_trajectory_from_grid(cost_map, start_node, end_node, smooth=True, s=50, k=2, num_points=20):
    path_finder = PathFinder(cost_map)
    path = path_finder.find_path(start_node, end_node)

    if smooth and len(path) > 2:
        return SmoothPath(path, s=s, k=k, num_points=num_points).path
    return path
