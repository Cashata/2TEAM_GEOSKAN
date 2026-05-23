from heapq import heappop, heappush

import numpy as np


class PathFinder:
    _NEIGHBORS = (
        (-1, -1, 1414),
        (0, -1, 1000),
        (1, -1, 1414),
        (-1, 0, 1000),
        (1, 0, 1000),
        (-1, 1, 1414),
        (0, 1, 1000),
        (1, 1, 1414),
    )

    def __init__(self, cost_map):
        self.cost_map = cost_map.T
        self.cardinal = 1000
        self.diagonal = 1414
        self.greed = 1
        positive = self.cost_map[self.cost_map > 0]
        self._min_cost = float(positive.min()) if positive.size else 1.0

    def find_path(self, start, end):
        return self._contraction(self._path_to(start, end))

    def _path_to(self, start, end):
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))

        if not self._is_passable(start) or not self._is_passable(end):
            return np.empty((0, 2), dtype=np.uint16)

        open_heap = []
        counter = 0
        start_score = 0.0
        heappush(open_heap, (self._heuristic(start, end), counter, start))
        came_from = {}
        g_score = {start: start_score}
        closed = set()

        while open_heap:
            _, _, current = heappop(open_heap)
            if current in closed:
                continue
            if current == end:
                return self._reconstruct_path(came_from, current)

            closed.add(current)
            current_score = g_score[current]
            for neighbor, step_cost in self._iter_neighbors(current):
                if neighbor in closed:
                    continue
                next_score = current_score + step_cost
                if next_score >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = next_score
                counter += 1
                priority = next_score + self.greed * self._heuristic(neighbor, end)
                heappush(open_heap, (priority, counter, neighbor))

        return np.empty((0, 2), dtype=np.uint16)

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return np.array(path, dtype=np.uint16)

    def _iter_neighbors(self, node):
        x, y = node
        for dx, dy, base_cost in self._NEIGHBORS:
            neighbor = (x + dx, y + dy)
            if not self._is_passable(neighbor):
                continue
            yield neighbor, base_cost * float(self.cost_map[neighbor])

    def _heuristic(self, node, end):
        dx = abs(end[0] - node[0])
        dy = abs(end[1] - node[1])
        diagonal_steps = min(dx, dy)
        cardinal_steps = max(dx, dy) - diagonal_steps
        return self._min_cost * (self.diagonal * diagonal_steps + self.cardinal * cardinal_steps)

    def _is_passable(self, node):
        x, y = node
        return 0 <= x < self.cost_map.shape[0] and 0 <= y < self.cost_map.shape[1] and self.cost_map[x, y] > 0

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
        for x, y in self._bresenham(start, end):
            if not self._is_passable((x, y)):
                return False
        return True

    def _bresenham(self, start, end):
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                error += dy
                x0 += sx
            if doubled_error <= dx:
                error += dx
                y0 += sy
