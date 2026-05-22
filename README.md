# 2TEAM_GEOSKAN

Практичный контур для задания "Дроно-старт": полет Geoscan Pioneer Mini 2 по локальным waypoint, локализация кадра на карте 3 x 3 м через ORB/RANSAC homography, распознавание целевых ArUco ID `3`, `23`, `42`, `117` и восстановление слова `ИТМО`.

## Модули

| Путь | Назначение |
| --- | --- |
| `fly_orb_ransac.py` | Совместимая CLI-обертка основного сценария. |
| `geoscan_mission/cli/fly_orb_ransac.py` | Сборка полетного/replay-сценария из модулей. |
| `geoscan_mission/flight/` | Управление дроном, SDK2/OpenCV/video camera adapters, battery check, waypoint-команды. |
| `geoscan_mission/vision/localization.py` | ORB/RANSAC-локализация кадра на reference-карте. |
| `geoscan_mission/vision/aruco.py` | ArUco-детектор целей миссии и накопление найденного слова. |
| `geoscan_mission/trajectory/` | Паттерны `waypoints`, `square`, `lawnmower`, `cube` и grid path helpers. |
| `geoscan_mission/recording.py` | CSV/JSON-логи, видеооверлеи, ArUco-summary и проекция маркеров на карту. |
| `keypoint_map_localizer.py` | Отдельная утилита для проверки RANSAC-локализации по изображению, видео или камере Mini 2. |
| `aruco` | CLI для проверки ArUco на одном изображении. |

Корневые `aruco_detector.py`, `PathFinder.py` и `SmoothPath.py` оставлены как compatibility re-export, чтобы старые импорты не ломались.

## Быстрый запуск

Локальная проверка с веб-камеры без взлета:

```bash
python fly_orb_ransac.py --no-flight --reference map.jpg --camera-index 0 --aruco
```

Запуск на Mini 2:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --aruco
```

Проверка одного изображения на ArUco:

```bash
python aruco --image frame.jpg --json --draw frame_aruco.jpg
```

## Проверка по видео с дрона

Да, записанное видео можно использовать без дрона. Это удобно для отладки зрения, RANSAC-локализации, ArUco-детекции, CSV-логов и overlay-видео.

Полный replay через основной пайплайн:

```bash
python fly_orb_ransac.py ^
  --no-flight ^
  --input-video flight.mp4 ^
  --reference map.jpg ^
  --aruco ^
  --csv replay_localization.csv ^
  --debug-dir debug/replay ^
  --video-camera-out replay_camera_overlay.avi ^
  --video-map-out replay_map_trace.avi ^
  --no-command-listener ^
  --no-flight-seconds 3600
```

`--no-flight-seconds` задает максимальную длительность replay. Если видео закончится раньше, обработка остановится по EOF сама.

Быстрая проверка только RANSAC-локализации:

```bash
python keypoint_map_localizer.py ^
  --reference map.jpg ^
  --video flight.mp4 ^
  --csv video_localization.csv ^
  --output-dir debug/localizer ^
  --frame-step 5
```

Для одного кадра:

```bash
python keypoint_map_localizer.py --reference map.jpg --image frame.jpg --output-dir debug/frame
```

## Логи ArUco

При включенном `--aruco` каждая JSON/CSV-строка получает дополнительные поля:

- `aruco_seen_ids` - ID, найденные в текущем кадре;
- `aruco_new_ids` - ID, впервые найденные за текущий запуск;
- `aruco_word` - слово из накопленных букв по возрастанию ID;
- `aruco_allowed_ids` и `aruco_forbidden_ids` - накопленные разрешенные и запрещенные цели;
- `aruco_markers_json` - JSON со списком маркеров, углами/центром в пикселях и, если homography валидна, координатами на карте.

Если ORB/RANSAC homography валидна, центр ArUco проецируется из пикселей кадра в координаты карты 3 x 3 м. Если homography невалидна, координаты карты остаются `null`, но pixel center/corners сохраняются.

## Траектории

Основной сценарий поддерживает:

- `--trajectory waypoints` - дефолтные точки или точки из повторяющихся `--waypoint x,y,z`;
- `--trajectory square` - квадрат по границе рабочей зоны с `--area-size`, `--margin`, `--height`;
- `--trajectory lawnmower` - змейка по сетке `--grid-size`;
- `--trajectory cube` - несколько слоев lawnmower по высотам `--layers` или `--height/--high-height`.

Пример:

```bash
python fly_orb_ransac.py --reference map.jpg --trajectory lawnmower --grid-size 4 --margin 0.25 --height 1.0 --aruco
```

## Разделение ответственности

- Менять команды дрона и safety-логику: `geoscan_mission/flight/control.py`.
- Менять источники кадров: `geoscan_mission/flight/camera.py`.
- Менять маршруты: `geoscan_mission/trajectory/patterns.py`.
- Менять ORB/RANSAC фильтры: `geoscan_mission/vision/localization.py`.
- Менять ArUco ID, буквы и типы целей: `geoscan_mission/vision/aruco.py`.
- Менять CSV/JSON/overlay: `geoscan_mission/recording.py`.

Файлы `repomix-output*.md` остаются локальными справочными дампами и не коммитятся.
