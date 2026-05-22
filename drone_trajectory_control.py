#!/usr/bin/env python3
"""
Скрипт управления дроном Геоскан Пионер Мини2 по траектории из grid_path.py.
Использует set_manual_speed_body_fixed для перемещения между точками,
и go_to_local_point_body_fixed для финального участка (предпоследняя -> последняя точка).

Учтены критические технические нюансы:
- Преобразование координат из глобальной системы в body-fixed
- Плавное снижение скорости при приближении к цели (PID-регулятор)
- Динамическое вычисление угла yaw по вектору движения
"""

from __future__ import annotations

import numpy as np
import time
import math

# Импортируем классы траектории из grid_path.py
from grid_path import PathFinder, SmoothPath


class PIDController:
    """PID-регулятор для плавного управления скоростью с устранением overshoot."""
    
    def __init__(self, kp=1.0, ki=0.02, kd=0.1, output_limit=1.5):
        self.kp = kp  # Пропорциональный коэффициент
        self.ki = ki  # Интегральный коэффициент
        self.kd = kd  # Дифференциальный коэффициент
        self.output_limit = output_limit  # Ограничение выходного сигнала
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_call = True
    
    def reset(self):
        """Сброс состояния регулятора."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_call = True
    
    def compute(self, error, dt):
        """
        Вычисление управляющего сигнала PID-регулятора.
        
        Args:
            error: Ошибка (расстояние до цели)
            dt: Время между вызовами (сек)
        
        Returns:
            Управляющий сигнал (скорость)
        """
        if dt <= 0:
            dt = 0.05  # Защита от деления на ноль
        
        # Пропорциональная составляющая
        p_term = self.kp * error
        
        # Интегральная составляющая (с анти-windup)
        self.integral += error * dt
        # Ограничение интеграла для предотвращения windup
        integral_limit = self.output_limit / (self.ki + 0.001) if self.ki > 0 else 10.0
        self.integral = max(-integral_limit, min(integral_limit, self.integral))
        i_term = self.ki * self.integral
        
        # Дифференциальная составляющая (скорость изменения ошибки)
        if self.first_call:
            d_term = 0.0
            self.first_call = False
        else:
            derivative = (error - self.previous_error) / dt
            d_term = self.kd * derivative
        
        self.previous_error = error
        
        # Суммарный управляющий сигнал
        output = p_term + i_term + d_term
        
        # Ограничение выходного сигнала
        output = max(-self.output_limit, min(self.output_limit, output))
        
        # Скорость не может быть отрицательной (движемся только вперёд к цели)
        return max(0.0, output)


class DroneController:
    """Контроллер дрона Геоскан Пионер Мини2."""

    def __init__(self, pioneer):
        """
        Инициализация контроллера.
        
        Args:
            pioneer: Объект дрона Pioneer с методами управления
        """
        self.p = pioneer
        # Максимальная скорость перемещения (м/с)
        self.max_speed = 1.5
        # Допуск достижения точки (м)
        self.position_tolerance = 0.1
        # Допуск для перехода на финальный участок (м)
        self.approach_tolerance = 0.3
        
        # PID-регуляторы для каждой оси
        self.pid_x = PIDController(kp=1.2, ki=0.03, kd=0.15, output_limit=self.max_speed)
        self.pid_y = PIDController(kp=1.2, ki=0.03, kd=0.15, output_limit=self.max_speed)
        self.pid_z = PIDController(kp=1.0, ki=0.02, kd=0.1, output_limit=self.max_speed)
        self.pid_yaw = PIDController(kp=2.0, ki=0.05, kd=0.3, output_limit=1.0)
        
        # Шаг времени (20 Гц)
        self.dt = 0.05

    def get_position(self):
        """Получить текущую позицию дрона в локальной системе координат."""
        # Предполагаем, что дрон предоставляет метод получения позиции
        # В реальной реализации используйте соответствующий метод API
        return self.p.get_position() if hasattr(self.p, 'get_position') else (0, 0, 0)

    def get_yaw(self):
        """Получить текущий угол рыскания дрона (рад)."""
        # Предполагаем, что дрон предоставляет метод получения yaw
        # В реальной реализации используйте соответствующий метод API
        if hasattr(self.p, 'get_telemetry'):
            telemetry = self.p.get_telemetry()
            if telemetry and 'yaw' in telemetry:
                return telemetry['yaw']
        return 0.0

    def transform_global_to_body_fixed(self, vx_global, vy_global, yaw):
        """
        Преобразовать вектор скорости из глобальной системы координат в body-fixed.
        
        Формула поворота на угол yaw:
        vx_body = vx_global·cos(yaw) + vy_global·sin(yaw)
        vy_body = -vx_global·sin(yaw) + vy_global·cos(yaw)
        
        Args:
            vx_global: Скорость по оси X в глобальной системе
            vy_global: Скорость по оси Y в глобальной системе
            yaw: Текущий угол рыскания дрона (рад)
            
        Returns:
            (vx_body, vy_body): Вектор скорости в body-fixed системе
        """
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        vx_body = vx_global * cos_yaw + vy_global * sin_yaw
        vy_body = -vx_global * sin_yaw + vy_global * cos_yaw
        
        return vx_body, vy_body

    def move_to_point_with_speed(self, target_x, target_y, target_z, current_yaw=None):
        """
        Перемещение к точке с помощью управления вектором скорости.
        Использует set_manual_speed_body_fixed(vx, vy, vz, yaw).
        
        Реализует PID-регулятор для каждой оси с плавным снижением скорости.
        Преобразует вектор скорости из глобальной системы в body-fixed.
        
        Args:
            target_x: Целевая координата X (м)
            target_y: Целевая координата Y (м)
            target_z: Целевая координата Z (м)
            current_yaw: Текущий угол рыскания (если None, будет получен от дрона)
            
        Returns:
            bool: True если точка достигнута, False иначе
        """
        current_x, current_y, current_z = self.get_position()
        
        # Если yaw не передан, получаем текущий угол дрона
        if current_yaw is None:
            current_yaw = self.get_yaw()
        
        # Вычисляем ошибки по каждой оси (расстояние до цели)
        error_x = target_x - current_x
        error_y = target_y - current_y
        error_z = target_z - current_z
        
        # Вычисляем горизонтальное расстояние до цели (для проверки достижения и yaw)
        horizontal_distance = np.sqrt(error_x**2 + error_y**2)
        full_distance = np.sqrt(horizontal_distance**2 + error_z**2)
        
        if full_distance < self.position_tolerance:
            # Точка достигнута, останавливаем дрон
            self.p.set_manual_speed_body_fixed(0, 0, 0, current_yaw)
            # Сбрасываем PID-регуляторы для следующего участка
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            self.pid_yaw.reset()
            return True
        
        # Вычисляем желаемый угол yaw по направлению движения
        desired_yaw = math.atan2(error_y, error_x) if horizontal_distance > 0.01 else current_yaw
        
        # Вычисляем ошибку по yaw (нормализуем угол в диапазон [-pi, pi])
        yaw_error = desired_yaw - current_yaw
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        # PID-регуляторы вычисляют скорость для каждой оси
        vx_global = self.pid_x.compute(error_x, self.dt)
        vy_global = self.pid_y.compute(error_y, self.dt)
        vz_global = self.pid_z.compute(error_z, self.dt)
        yaw_rate = self.pid_yaw.compute(abs(yaw_error), self.dt)
        
        # Определяем направление вращения для yaw
        if yaw_error < 0:
            yaw_rate = -yaw_rate
        
        # Преобразуем вектор скорости из глобальной системы в body-fixed
        vx_body, vy_body = self.transform_global_to_body_fixed(vx_global, vy_global, current_yaw)
        
        # Устанавливаем скорость через set_manual_speed_body_fixed
        # yaw передаем как желаемый угол ориентации по направлению движения
        self.p.set_manual_speed_body_fixed(vx_body, vy_body, vz_global, desired_yaw)
        
        return False

    def fly_trajectory(self, path_points, z_height=1.0):
        """
        Полет по траектории из точек.
        
        Args:
            path_points: Массив точек траектории [[x1, y1], [x2, y2], ...]
            z_height: Высота полета (м)
        """
        if len(path_points) < 2:
            print("Траектория должна содержать минимум 2 точки")
            return
        
        print(f"Начало полета по траектории из {len(path_points)} точек")
        
        # Проходим по всем точкам кроме последней двух
        for i in range(len(path_points) - 2):
            point = path_points[i]
            target_x, target_y = point[0], point[1]
            
            print(f"Перемещение к точке {i+1}: ({target_x:.2f}, {target_y:.2f}, {z_height:.2f})")
            
            # Используем set_manual_speed_body_fixed для перемещения
            while not self.move_to_point_with_speed(target_x, target_y, z_height):
                time.sleep(0.05)  # Частота обновления 20 Гц для плавности
            
            # Короткая пауза перед следующей точкой
            time.sleep(0.3)
        
        # Предпоследняя точка - подготовка к финальному участку
        second_last_point = path_points[-2]
        last_point = path_points[-1]
        
        print(f"Подход к предпоследней точке: ({second_last_point[0]:.2f}, {second_last_point[1]:.2f})")
        
        # Сначала долетаем до предпоследней точки с замедлением
        while not self.move_to_point_with_speed(
            second_last_point[0], 
            second_last_point[1], 
            z_height
        ):
            time.sleep(0.05)
        
        time.sleep(0.3)
        
        # Финальный участок: из предпоследней точки в последнюю
        # Используем go_to_local_point_body_fixed как требуется
        print(f"Финальный участок: от ({second_last_point[0]:.2f}, {second_last_point[1]:.2f}) "
              f"к ({last_point[0]:.2f}, {last_point[1]:.2f})")
        
        # Вычисляем относительные координаты для go_to_local_point_body_fixed
        # (относительно текущей позиции дрона в предпоследней точке)
        current_x, current_y, current_z = self.get_position()
        relative_x = last_point[0] - current_x
        relative_y = last_point[1] - current_y
        relative_z = z_height - current_z
        
        # Вычисляем yaw для финальной точки по направлению движения
        final_yaw = math.atan2(last_point[1] - second_last_point[1], 
                               last_point[0] - second_last_point[0])
        
        # Выполняем финальное перемещение через go_to_local_point_body_fixed
        self.p.go_to_local_point_body_fixed(relative_x, relative_y, relative_z, final_yaw)
        
        # Ждем достижения последней точки
        print("Ожидание достижения финальной точки...")
        while True:
            current_x, current_y, current_z = self.get_position()
            distance = np.sqrt((last_point[0] - current_x)**2 + 
                              (last_point[1] - current_y)**2 + 
                              (z_height - current_z)**2)
            
            if distance < self.position_tolerance:
                break
            
            time.sleep(0.1)
        
        print("Траектория выполнена!")
        
        # Останавливаем дрон
        self.p.set_manual_speed_body_fixed(0, 0, 0, final_yaw)


def create_trajectory_from_grid(cost_map, start_node, end_node, smooth=True):
    """
    Создать траекторию на основе карты стоимости.
    
    Args:
        cost_map: Карта стоимости для PathFinder
        start_node: Начальная точка [x, y]
        end_node: Конечная точка [x, y]
        smooth: Сглаживать ли траекторию
    
    Returns:
        Массив точек траектории
    """
    # Создаем поисковик пути
    path_finder = PathFinder(cost_map)
    
    # Находим путь
    path = path_finder.find_path(start_node, end_node)
    
    if smooth and len(path) > 2:
        # Сглаживаем путь
        smooth_path = SmoothPath(path, s=50, k=2, num_points=20)
        return smooth_path.path
    else:
        return path


def main():
    """Основная функция демонстрации."""
    # Пример использования
    print("Геоскан Пионер Мини2 - Управление по траектории")
    print("=" * 50)
    
    # Создаем пример карты стоимости (в реальности загрузите свои данные)
    cost_map = np.ones((50, 50), dtype=np.uint16) * 100  # Карта 50x50 с базовой стоимостью 100
    
    # Определяем начало и конец пути
    start = [5, 5]
    end = [45, 45]
    
    # Генерируем траекторию
    trajectory = create_trajectory_from_grid(cost_map, start, end, smooth=True)
    
    print(f"Сгенерирована траектория из {len(trajectory)} точек")
    print("Первые 5 точек:", trajectory[:5])
    print("Последние 5 точек:", trajectory[-5:])
    
    # Примечание: Для реального полета необходимо:
    # 1. Подключиться к дрону
    # 2. Передать объект pioneer в DroneController
    # 3. Вызвать controller.fly_trajectory(trajectory)
    
    print("\nДля запуска реального полета:")
    print("1. Инициализируйте подключение к дрону Pioneer")
    print("2. Создайте экземпляр: controller = DroneController(pioneer)")
    print("3. Запустите полет: controller.fly_trajectory(trajectory, z_height=1.0)")


if __name__ == "__main__":
    main()
