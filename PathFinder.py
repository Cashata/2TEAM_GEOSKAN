import numpy as np
import tcod
from tcod.path import SimpleGraph, Pathfinder


class PathFinder:
    def __init__(self, cost_map):
        self.cost_map = cost_map.T
        self.graph = SimpleGraph(cost=self.cost_map, cardinal=1000, diagonal=1414, greed=1)

    def find_path(self, start, end):
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
        for x, y in tcod.los.bresenham(start, end):
            if self.cost_map[x, y] <= 0:
                return False
        return True
