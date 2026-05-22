# 2TEAM_GEOSKAN

## Текущий MVP

Репозиторий собирает практичный контур для задания "Дроно-старт": полет Geoscan Pioneer Mini 2 по локальным waypoint, локализация кадра на карте 3 x 3 м через ORB/RANSAC homography, распознавание целевых ArUco ID `3`, `23`, `42`, `117` и восстановление слова `ИТМО`.

Ключевые файлы и модули:

| Файл | Назначение |
| --- | --- |
| `fly_orb_ransac.py` | Совместимая CLI-обертка для основного сценария. Старые команды запуска остаются рабочими. |
| `geoscan_mission/flight/` | Управление дроном, SDK2 camera adapters, battery check, waypoint-команды и ожидание достижения точки. |
| `geoscan_mission/vision/` | ORB/RANSAC-локализация по карте и ArUco-детектор целей миссии. |
| `geoscan_mission/trajectory/` | Генерация маршрутов `waypoints/square/lawnmower/cube` и grid path planning helpers. |
| `geoscan_mission/recording.py` | Непрерывная запись кадров, CSV/JSON-строки, видеооверлеи и ArUco-проекции на карту. |
| `aruco_detector.py` | Совместимый re-export ArUco-детектора для старых импортов. |
| `aruco` | CLI-обертка для проверки одного изображения: `python aruco --image frame.jpg --json`. |
| `collect_dataset_mini2.py` | Автосбор кадров с Mini 2 для датасета landmarks/YOLO. |
| `keypoint_map_localizer.py` | Отдельная утилита для проверки локализации по изображению, видео или камере Mini 2. |
| `drone_start_assignment_llm.md` | Машиночитаемая версия задания с критериями, словарем ArUco и чеклистом команды. |

Быстрый локальный тест без взлета:

```bash
python fly_orb_ransac.py --no-flight --reference map.jpg --camera-index 0 --aruco
```

Запуск на Mini 2:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --aruco
```

## Как отдельно дорабатывать части системы

Новая структура сделана так, чтобы команда могла править отдельные слои без конфликтов:

- управление полетом: `geoscan_mission/flight/control.py`;
- камеры: `geoscan_mission/flight/camera.py`;
- маршруты и траектории: `geoscan_mission/trajectory/patterns.py`;
- grid path planning из старых `PathFinder.py`/`SmoothPath.py`: `geoscan_mission/trajectory/grid_path.py`;
- ArUco: `geoscan_mission/vision/aruco.py`;
- ORB/RANSAC-локализация: `geoscan_mission/vision/localization.py`;
- логи, CSV, JSON и видеооверлей: `geoscan_mission/recording.py`;
- сборка CLI-сценариев: `geoscan_mission/cli/`.

Корневые `fly_orb_ransac.py`, `aruco`, `aruco_detector.py`, `PathFinder.py` и `SmoothPath.py` оставлены как compatibility wrappers, чтобы старые команды и импорты не ломались.

При включенном `--aruco` каждая JSON/CSV-строка получает дополнительные поля:

- `aruco_seen_ids` - ID, найденные в текущем кадре;
- `aruco_new_ids` - ID, впервые найденные за текущий запуск;
- `aruco_word` - слово из накопленных букв по возрастанию ID;
- `aruco_allowed_ids` и `aruco_forbidden_ids` - накопленные разрешенные и запрещенные цели;
- `aruco_markers_json` - JSON со списком маркеров, углами/центром в пикселях и, если homography валидна, координатами на карте.

Файлы `repomix-output*.md` используются как локальные справочные дампы полезных репозиториев и примеров: ArUco-полет, патруль по сетке, camera calibration, detection API, cargo-подходы. В Git они намеренно не добавляются, чтобы не раздувать историю, особенно из-за большого `repomix-output.md`.

## Справочные материалы по CV/ML-архитектуре

Проект рассчитан на задачу с дроном Geoscan Pioneer Mini 2 над плоским 2D-ландшафтом примерно 3 x 3 м. На карте могут быть размечены база, зоны поиска, запретные зоны и ArUco-цели. Рекомендуемая идея: использовать YOLO для поиска визуальных объектов карты, а OpenCV ArUco для точного чтения ID целей.

Базовый пайплайн:

```text
камера Mini 2
  -> YOLO: объекты карты, запретные зоны, база, кандидаты целей
  -> homography: перевод пикселей в координаты карты 3 x 3 м
  -> OpenCV ArUco: точный ID цели
  -> planner: маршрут с обходом no-fly zones
  -> Pioneer-SDK2: полет, захват, возврат на базу
```

## Основные библиотеки и инструменты

| Инструмент | Для чего нужен |
| --- | --- |
| `Pioneer-SDK2` | Управление Mini 2: взлет, посадка, полет в локальную точку, проверка достижения waypoint, работа с полезной нагрузкой и захватом. |
| `Camera` из SDK | Получение BGR-кадров с камеры Mini 2 для обработки OpenCV/YOLO. |
| `ImageViewer` | Трансляция обработанного CV-кадра в браузер, чтобы показывать boxes, ID, зоны и маршрут. |
| `Pioneer-RKNN` | Запуск нейросетей на Mini 2 через RKNN. Подходит для YOLO-моделей, загруженных в формате `.rknn`. |
| `OpenCV ArUco` | Детектирование ArUco-маркеров и чтение их ID. Лучше использовать для точного определения цели, а не обучать YOLO читать ID. |
| `OpenCV findHomography` | Перевод координат между плоскостью изображения и плоскостью карты 3 x 3 м. |
| `Ultralytics YOLO` | Обучение легкой модели YOLOv8n/YOLO11n на ноутбуке перед конвертацией в RKNN. |
| `CVAT`, `Roboflow`, `Label Studio` | Разметка датасета в YOLO-формате. |

## Что должен делать YOLO

YOLO лучше использовать как детектор объектов карты: он отвечает на вопрос "что и где находится на ландшафте". Точный ID цели лучше читать через ArUco.

Возможные классы для первой версии:

```yaml
names:
  0: corner_tl
  1: corner_tr
  2: corner_br
  3: corner_bl
  4: base
  5: no_fly_zone
  6: search_zone
  7: road_crossing
  8: target_candidate
```

Углы карты полезно размечать отдельными классами (`corner_tl`, `corner_tr`, `corner_br`, `corner_bl`), чтобы модель сразу понимала, какой угол какой. Это упрощает построение homography.

## Что должен делать ArUco

ArUco-маркеры нужны для точного определения ID цели. YOLO может находить область-кандидат (`target_candidate`), после чего crop передается в OpenCV ArUco.

Пример логики целей:

```python
TARGETS = {
    3:   {"type": "allowed",   "letter": "И"},
    23:  {"type": "allowed",   "letter": "Т"},
    42:  {"type": "forbidden", "letter": "М"},
    117: {"type": "forbidden", "letter": "О"},
}
```

Итоговое слово можно собирать по возрастанию ID найденных целей.

## Локализация на карте

### Вариант 1: homography по углам карты

На карте задаются четыре угла в метрах:

```text
(0.0, 0.0)
(3.0, 0.0)
(3.0, 3.0)
(0.0, 3.0)
```

YOLO или служебные ArUco находят эти углы в пикселях. Затем `cv2.findHomography()` строит преобразование "пиксель камеры -> координата карты в метрах".

Плюсы: простой и надежный вариант для плоской карты.

Минусы: нужно видеть достаточно опорных точек.

### Вариант 2: YOLO landmarks

YOLO обучается находить устойчивые ориентиры: углы карты, базу, перекрестки, зоны, характерные элементы ландшафта. У каждого ориентира заранее известны координаты на карте.

Плюсы: может работать при частичной видимости карты.

Минусы: нужен датасет и аккуратная разметка.

## Запретные зоны

### Вариант 1: зоны из конфига

Если координаты известны заранее, их проще хранить как полигоны в mission-конфиге. Планировщик расширяет полигоны на safety margin и строит путь вокруг.

### Вариант 2: YOLO-детекция no-fly zones

Если запретные зоны нанесены на карту визуально, YOLO может находить класс `no_fly_zone`. Bounding box переводится через homography в метры, затем зона расширяется на safety margin.

Для Mini 2 практичнее начинать с detection-подхода. Segmentation/OBB может быть точнее, но сложнее для быстрого MVP.

## Планирование маршрута

### Вариант 1: A* по сетке

Карта 3 x 3 м делится на сетку, например 5 см:

```text
3 м / 0.05 м = 60 клеток
итого 60 x 60
```

Запретные зоны отмечаются как занятые, затем A* строит маршрут. После построения путь желательно сгладить.

### Вариант 2: visibility graph / waypoint graph

Граф строится из базы, целей, углов зон поиска и углов запретных зон с отступом. Ребро разрешено, если линия между точками не пересекает no-fly polygon.

## Стратегия поиска

Для соревнования проще всего начать с разбиения карты на зоны и полета "змейкой". Например, 3 x 3 м делятся на 4-9 областей, каждый дрон получает свою область, а внутри области проходит параллельными линиями с фиксированным шагом.

Более продвинутый вариант - вероятностная карта поиска: у каждой клетки есть score, который повышается при обнаружении `target_candidate` и понижается после просмотра зоны.

## Координация нескольких дронов

Для MVP лучше использовать центральный координатор на ноутбуке. Каждый дрон отправляет состояние:

```json
{
  "drone_id": "mini2_1",
  "xy": [1.25, 2.10],
  "state": "searching",
  "found": [{"id": 23, "xy": [2.4, 1.1]}],
  "cargo": false
}
```

Координатор распределяет зоны, хранит найденные цели, решает конфликты и собирает итоговое слово.

Распределенная координация между дронами возможна, но сложнее: нужно обмениваться сообщениями о найденных целях, занятых зонах и доставленных объектах.

## Захват и возврат на базу

Простой вариант: после определения координаты цели дрон летит в точку, снижается, закрывает захват, возвращается на базу и открывает захват.

Более надежный вариант: перед захватом включить visual servoing по ArUco. В финальной фазе дрон центрирует маркер в кадре, снижает ошибку до threshold и только потом выполняет захват.

Базу можно задавать фиксированной координатой или размечать визуальным маркером `base`/ArUco для точного захода.

## Датасет для YOLO landmarks

Датасет для `YOLO landmarks` нужен не для чтения ID ArUco, а для надежного поиска устойчивых объектов карты: углов, базы, запретных зон, зон поиска, перекрестков и кандидатов целей. YOLO в этой схеме отвечает за вопрос "где на изображении находится объект карты", а точную идентификацию цели лучше оставлять OpenCV ArUco.

Для старта можно вручную собрать и разметить 200-500 кадров. Но если запустить автосбор с Mini 2 по разным высотам, позициям и yaw, реально получить 500-1500 кадров без ручной съемки. После первой версии модели обычно нужно доразметить только ошибки: плохой свет, смаз, частичную видимость, тени и похожие элементы ландшафта.

Снимать нужно не "красивые" кадры, а разные условия, в которых модель потом будет работать:

- высоты: 0.6 / 0.8 / 1.0 / 1.2 м;
- разные yaw-углы: 0 / 45 / 90 / 135 / 180 / 225 / 270 / 315 градусов;
- разный свет;
- частичная видимость карты;
- кадры с ArUco и без ArUco;
- кадры, где видна только часть no-fly зоны или базы;
- кадры с тенями, руками, другими дронами и визуальными помехами.

Структура датасета для YOLO:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Пример `data.yaml`:

```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: corner_tl
  1: corner_tr
  2: corner_br
  3: corner_bl
  4: base
  5: no_fly_zone
  6: target_candidate
```

## Автоматизация сбора датасета

Сбор кадров можно почти полностью автоматизировать скриптом на Mini 2. Дрон сам взлетает, проходит сетку точек над картой, меняет высоту и yaw, а камера сохраняет JPG-кадры. Для этого нужны:

- `Pioneer` для управления дроном;
- `Camera` и `CameraType.OPT` для получения BGR-кадра с нижней/оптической камеры;
- `ServoCamera`, если нужно повернуть камеру вниз;
- `go_to_local_point(x, y, z, yaw)` для перехода в точку;
- `point_reached()` для ожидания достижения waypoint;
- `Camera.get_cv_frame()` для получения кадра;
- `cv2.imwrite()` для сохранения JPG.

Сохранять лучше сразу изображения, а не только видео: для YOLO все равно нужны пары `image + label`, а кадры из автополета проще сразу складывать в `images_raw`.

Пример скрипта автосъемки:

```python
import os
import time
import cv2
from pioneer_sdk2 import Pioneer, Camera, CameraType, ServoCamera

OUT_DIR = "/home/geoscan/dataset_landmarks/images_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# Точки задаются относительно места взлета.
HEIGHTS = [0.6, 0.8, 1.0, 1.2]
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]

XY_POINTS = [
    (0.0, 0.0),
    (0.5, 0.0),
    (-0.5, 0.0),
    (0.0, 0.5),
    (0.0, -0.5),
    (0.5, 0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (-0.5, -0.5),
]

def wait_point(pioneer, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pioneer.point_reached():
            return True
        time.sleep(0.1)
    return False

def save_frames(camera, prefix, seconds=3, fps=2):
    delay = 1.0 / fps
    n = int(seconds * fps)

    for i in range(n):
        frame = camera.get_cv_frame(timeout=2.0)
        if frame is not None:
            path = f"{OUT_DIR}/{prefix}_{i:03d}.jpg"
            cv2.imwrite(path, frame)
        time.sleep(delay)

pioneer = Pioneer()
camera = Camera(camera_type=CameraType.OPT)

# Если сервопривод камеры доступен, направляем камеру вниз.
try:
    servo = ServoCamera()
    servo.set_angle(-90)
except Exception as e:
    print("ServoCamera недоступна или не настроена:", e)

try:
    pioneer.arm()
    pioneer.takeoff()

    idx = 0
    for z in HEIGHTS:
        for x, y in XY_POINTS:
            for yaw in YAWS:
                pioneer.go_to_local_point(x=x, y=y, z=z, yaw=yaw)
                wait_point(pioneer, timeout=12)
                time.sleep(0.5)

                prefix = f"z{z:.1f}_x{x:.1f}_y{y:.1f}_yaw{yaw:03d}_{idx:05d}"
                save_frames(camera, prefix, seconds=2, fps=2)
                idx += 1

    pioneer.land()

except KeyboardInterrupt:
    print("Остановлено вручную, посадка")
    pioneer.land()

finally:
    camera.stop()
```

Такой маршрут дает:

- 4 высоты;
- 9 позиций вокруг точки взлета;
- 8 yaw-углов;
- несколько кадров в каждой точке.

Даже при 2 секундах съемки и 2 FPS получается больше тысячи изображений. Потом их можно разложить на `train/val`, разметить руками или прогнать через авторазметку ниже.

### Авторазметка через ArUco-углы карты

Для карты 3 x 3 м полезно автоматизировать не только съемку, но и большую часть разметки. Идея такая:

```text
служебные ArUco по углам карты
  -> cv2.aruco.detectMarkers()
  -> homography: координаты карты 3 x 3 м -> пиксели кадра
  -> известные LANDMARKS в метрах
  -> pixel boxes
  -> YOLO .txt labels
```

На 4 угла карты кладутся служебные ArUco, например ID `10`, `11`, `12`, `13`. Они не являются целями миссии и нужны только для автосбора датасета. OpenCV ArUco находит их в кадре, `cv2.findHomography()` строит преобразование из координат карты в пиксели, а заранее заданные объекты карты проецируются в bounding boxes.

Пример объектов карты в метрах:

```python
LANDMARKS = [
    # class_id, x_min, y_min, x_max, y_max на карте 3 x 3 м
    (0, 0.00, 0.00, 0.15, 0.15),  # corner_tl
    (1, 2.85, 0.00, 3.00, 0.15),  # corner_tr
    (2, 2.85, 2.85, 3.00, 3.00),  # corner_br
    (3, 0.00, 2.85, 0.15, 3.00),  # corner_bl
    (4, 0.20, 0.20, 0.55, 0.55),  # base
    (5, 1.20, 1.00, 1.80, 1.60),  # no_fly_zone
]
```

Пример генерации YOLO-labels:

```python
import cv2
import numpy as np

MAP_CORNERS_M = {
    10: (0.0, 0.0),
    11: (3.0, 0.0),
    12: (3.0, 3.0),
    13: (0.0, 3.0),
}

def yolo_line_from_pixel_box(class_id, x1, y1, x2, y2, W, H):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H

    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def project_box(H_map_to_img, box_m):
    class_id, x1, y1, x2, y2 = box_m
    pts = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]).reshape(-1, 1, 2)

    pix = cv2.perspectiveTransform(pts, H_map_to_img).reshape(-1, 2)
    px1, py1 = pix.min(axis=0)
    px2, py2 = pix.max(axis=0)

    return class_id, px1, py1, px2, py2

def make_labels_for_frame(frame, landmarks):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    img_pts = []
    map_pts = []

    for c, marker_id in zip(corners, ids.flatten()):
        if marker_id in MAP_CORNERS_M:
            center_px = c[0].mean(axis=0)
            img_pts.append(center_px)
            map_pts.append(MAP_CORNERS_M[marker_id])

    if len(img_pts) < 4:
        return None

    H_map_to_img, _ = cv2.findHomography(
        np.float32(map_pts),
        np.float32(img_pts),
    )

    if H_map_to_img is None:
        return None

    H_img, W_img = frame.shape[:2]
    labels = []

    for box_m in landmarks:
        class_id, px1, py1, px2, py2 = project_box(H_map_to_img, box_m)
        label = yolo_line_from_pixel_box(
            class_id,
            px1,
            py1,
            px2,
            py2,
            W_img,
            H_img,
        )
        if label is not None:
            labels.append(label)

    return labels
```

YOLO-строка имеет стандартный формат:

```text
class_id x_center y_center width height
```

Все координаты нормализуются от 0 до 1 относительно ширины и высоты изображения. После авторазметки нужно обязательно просмотреть часть результата: если один из угловых ArUco найден неправильно или кадр слишком смазан, labels лучше удалить или поправить вручную.

Практичный пайплайн:

1. Наклеить 4 служебных ArUco по углам карты.
2. Запустить автополет на разных высотах и yaw.
3. Сохранять сразу JPG-кадры.
4. Авторазметить landmarks через homography.
5. Проверить руками 10-15% кадров.
6. Удалить или поправить ошибочные кадры.
7. Обучить YOLO landmarks.
8. Конвертировать модель в `.rknn` и загрузить на Mini 2.

## Обучение и перенос на Mini 2

Обучать модель лучше на ноутбуке/ПК, а не на дроне. Для Mini 2 стоит брать маленькие модели: YOLOv8n или YOLO11n.

Типовая команда обучения:

```bash
pip install ultralytics
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640
```

После обучения модель нужно конвертировать в `.rknn`, затем загрузить на Mini 2 через сервис моделей ИИ. Для запуска на борту используется `Pioneer-RKNN`, например класс `Yolo` с именем загруженной модели.

## Рекомендуемая MVP-архитектура

```text
Mini 2 onboard:
  Camera
  YOLO RKNN: base / no_fly_zone / search_zone / target_candidate / map_corner
  OpenCV ArUco: ID 3, 23, 42, 117
  Homography: pixel -> map 3 x 3 m
  A*: route planning
  Pioneer-SDK2: go_to_local_point / grab / land
  ImageViewer: CV stream

Laptop coordinator:
  mission state
  zones assignment
  found targets
  final word
  multi-drone conflict control
```

Главное разделение ответственности: YOLO определяет объекты и зоны на ландшафте, ArUco подтверждает точный ID цели, а планировщик строит маршрут в координатах карты.
