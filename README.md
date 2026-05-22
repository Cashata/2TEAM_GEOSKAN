# 2TEAM_GEOSKAN

Практичный контур для задания "Дроно-старт": полет Geoscan Pioneer Mini 2 по локальным waypoint, локализация кадра на карте 3 x 3 м через ORB/RANSAC homography, распознавание целевых ArUco ID `3`, `23`, `42`, `117` и восстановление слова `ИТМО`.

## Модули

| Путь | Назначение |
| --- | --- |
| `fly_orb_ransac.py` | Совместимая CLI-обертка основного сценария. |
| `geoscan_mission/cli/fly_orb_ransac.py` | Сборка полетного/replay-сценария из модулей. |
| `geoscan_mission/flight/` | Управление дроном, SDK2/OpenCV/video camera adapters, battery check, waypoint-команды. |
| `geoscan_mission/flight/trajectory_control.py` | Экспериментальное PID/manual-speed управление через `set_manual_speed_body_fixed`. |
| `geoscan_mission/vision/localization.py` | ORB/RANSAC-локализация кадра на reference-карте. |
| `geoscan_mission/vision/aruco.py` | ArUco-детектор целей миссии и накопление найденного слова. |
| `geoscan_mission/trajectory/` | Паттерны `waypoints`, `square`, `lawnmower`, `cube` и grid path helpers. |
| `geoscan_mission/recording.py` | CSV/JSON-логи, видеооверлеи, ArUco-summary и проекция маркеров на карту. |
| `tools/keypoint_map_localizer.py` | Отдельная утилита для проверки RANSAC-локализации по изображению, видео или камере Mini 2. |
| `tools/collect_dataset_mini2.py` | Утилита сбора кадров с Mini 2 и dry-run предпросмотра маршрута. |
| `tools/grid_path_demo.py` | Demo для grid path planning вокруг запретной зоны на карте. |
| `tools/convert_map_tif.py` | Конвертация большой `*.tif/*.tiff` карты в рабочий `map.jpg`. |
| `tools/wasd_flight.py` | Ручное WASD-управление Mini 2 через `set_manual_speed_body_fixed`. |
| `docs/` | Задание и курируемые заметки по полезным источникам/улучшениям. |
| `aruco` | CLI для проверки ArUco на одном изображении. |

Корневые `fly_orb_ransac.py`, `aruco`, `aruco_detector.py`, `PathFinder.py`, `SmoothPath.py` и `drone_trajectory_control.py` оставлены как compatibility launchers/re-export, чтобы старые команды и импорты не ломались.

## Быстрый запуск

Локальная проверка с веб-камеры без взлета:

```bash
python fly_orb_ransac.py --no-flight --reference map.jpg --camera-index 0 --aruco
```

Запуск на Mini 2:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --aruco
```

Калибровка камеры Mini 2 без GUI на борту:

```bash
python3 calibration.py --camera-source sdk2 --sdk2-camera-type OPT --max-frames 30 --capture-interval 1.0 --output data.yml --frames-dir calibration_frames --debug-dir calibration_debug
```

По умолчанию ожидается шахматная доска `6 x 9` внутренних углов; если распечатка другая, задайте `--board-cols` и `--board-rows`.

После этого основной сценарий можно запускать с выпрямлением кадра:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --calibration data.yml --aruco
```

Если кадры уже собраны, пересчитать коэффициенты можно без дрона:

```bash
python calibration.py --images calibration_frames --glob "*.jpg" --output data.yml --debug-dir calibration_debug
```

Экспериментальный режим ручного управления скоростью:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --trajectory square --area-size 1.0 --margin 0.2 --height 0.8 --speed 0.12 --move-timeout 20 --control-mode manual-speed --aruco
```

По умолчанию используется `--control-mode autopilot`: SDK сам ведет дрон через `go_to_local_point`. `manual-speed` отправляет частые команды `set_manual_speed_body_fixed`, требует рабочие `get_local_position_lps()` и `get_local_yaw_lps()`, поэтому сначала проверяйте его на маленьком квадрате и низкой скорости.

Во время реального полета консольный listener понимает `rtl` + Enter. Сначала отправляется встроенный SDK RTL (`rtl()`, `return_to_launch()` или `return_to_home()`, если метод есть), иначе используется fallback: возврат в локальную точку `x=0, y=0` на текущей высоте и посадка.

Конвертация карты из TIFF в JPEG:

```bash
python tools/convert_map_tif.py Карта_Дроно_старт.tif -o map.jpg --max-side 6000
```

Ручное WASD-управление:

```bash
python3 tools/wasd_flight.py --speed 0.12 --vertical-speed 0.10 --sdk2-camera-type OPT
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
python tools/keypoint_map_localizer.py ^
  --reference map.jpg ^
  --video flight.mp4 ^
  --csv video_localization.csv ^
  --output-dir debug/localizer ^
  --frame-step 5
```

Для одного кадра:

```bash
python tools/keypoint_map_localizer.py --reference map.jpg --image frame.jpg --output-dir debug/frame
```

Dry-run маршрута для сбора кадров:

```bash
python tools/collect_dataset_mini2.py --dry-run
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

Режим управления выбирается отдельно от формы маршрута: `--control-mode autopilot` для штатного SDK waypoint-полета или `--control-mode manual-speed` для PID-контроллера по локальной позиции LPS.

## Разделение ответственности

- Менять команды дрона и safety-логику: `geoscan_mission/flight/control.py`.
- Менять PID/manual-speed движение: `geoscan_mission/flight/trajectory_control.py`.
- Менять источники кадров: `geoscan_mission/flight/camera.py`.
- Менять маршруты: `geoscan_mission/trajectory/patterns.py`.
- Менять ORB/RANSAC фильтры: `geoscan_mission/vision/localization.py`.
- Менять ArUco ID, буквы и типы целей: `geoscan_mission/vision/aruco.py`.
- Менять CSV/JSON/overlay: `geoscan_mission/recording.py`.

Файлы `repomix-output*.md` остаются локальными справочными дампами и не коммитятся.
