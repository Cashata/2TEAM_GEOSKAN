# 2TEAM_GEOSKAN

Проект для хакатонного задания "Дроно-старт" на Geoscan Pioneer Mini 2:
полет по локальным точкам, ORB/RANSAC-локализация кадра на карте 3 x 3 м,
поиск ArUco-маркеров, запись логов/видео и отдельные утилиты для отладки без
полета.

Файлы, которые недавно были загружены прямо в корень (`PathFinder.py`,
`SmoothPath.py`, `main.py`, `orb.py`, `photo_test.py`, `test_path.py`),
перенесены в папку `path_orb_experiment/`.

## Структура проекта

| Путь | Что внутри |
| --- | --- |
| `geoscan_mission/` | Основной пакет проекта. Здесь лежит модульная реализация полета, зрения, траекторий и логирования. |
| `geoscan_mission/cli/` | CLI-сценарии: основной полет/replay, ArUco-проверка, калибровка камеры. |
| `geoscan_mission/flight/` | Работа с Pioneer SDK2, камерами, waypoint-командами, manual-speed управлением и safety-командами. |
| `geoscan_mission/vision/` | ORB/RANSAC-локализация, Grid-ORB, ArUco-детектор и загрузка калибровки. |
| `geoscan_mission/trajectory/` | Генерация маршрутов `waypoints`, `square`, `lawnmower`, `cube`, grid path planning и сглаживание. |
| `geoscan_mission/recording.py` | CSV/JSON-строки, события миссии, видеооверлеи и проекция ArUco на карту. |
| `tools/` | Отдельные утилиты для карты, датасета, локализации, grid path demo и WASD-управления. |
| `path_orb_experiment/` | Перенесенный свежий эксперимент: A* PathFinder, SmoothPath, ORB-локализация по `map-2.png`, тест пути, фото-тест и прямой полетный скрипт. |
| `docs/` | Описание задания и справочные заметки по текущей реализации. |
| `calibration_info/` | Сохраненный набор калибровочных данных/результатов. |
| `flights/` | Логи и артефакты реальных/тестовых запусков. |
| `out/`, `calibration_frames/`, `calibration_debug/` | Локальные выходные данные калибровки и отладки. |

## Корневые сценарии

| Файл | Назначение |
| --- | --- |
| `fly_orb_ransac.py` | Совместимая обертка основного модульного сценария `geoscan_mission.cli.fly_orb_ransac`. |
| `calibration.py` | Обертка CLI калибровки камеры. |
| `aruco` | CLI для проверки ArUco на одном изображении. |
| `aruco_detector.py` | Re-export ArUco-детектора для старых импортов. |
| `drone_trajectory_control.py` | Demo/compatibility wrapper для manual-speed trajectory control. |
| `fly_record_video.py` | Готовый профиль полета по маленькому квадрату с записью видео и логов. |
| `fly_find_id15.py` | Автономная миссия поиска ArUco ID 15, посадки на него, повторного взлета и возврата. |
| `aruco_tracker_standalone.py` | Самодостаточный трекер ArUco без зависимости от `geoscan_mission`. |
| `aruco_hand_check.py` | Live preview для ручной проверки ArUco через MJPEG-сервер. |
| `trajectory_builder_standalone.py` | Самодостаточное построение waypoint/spline-маршрутов без полета. |
| `motion_mechanics_standalone.py` | Исполнение готовой траектории из JSON/CSV через body-fixed speed-команды. |
| `sync_to_drone.ps1` | Упаковка проекта и загрузка на дрон по `scp/ssh`. |

## Утилиты

| Файл | Назначение |
| --- | --- |
| `tools/convert_map_tif.py` | Конвертация большой TIFF-карты в рабочий JPEG. |
| `tools/keypoint_map_localizer.py` | Проверка ORB/RANSAC-локализации по изображению, видео или камере. |
| `tools/collect_dataset_mini2.py` | Сбор кадров с Mini 2 и dry-run предпросмотр маршрута. |
| `tools/grid_path_demo.py` | Демонстрация grid path planning вокруг запретной зоны. |
| `tools/wasd_flight.py` | Ручное WASD-управление Mini 2 через `set_manual_speed_body_fixed`. |

## Экспериментальная папка `path_orb_experiment`

| Файл | Назначение |
| --- | --- |
| `path_orb_experiment/PathFinder.py` | A* поиск пути по cost map с последующим сжатием ломаной. |
| `path_orb_experiment/SmoothPath.py` | Сглаживание пути через `scipy.interpolate.splprep/splev`. |
| `path_orb_experiment/orb.py` | ORB-детектор, который ищет положение кадра на карте `map-2.png`. |
| `path_orb_experiment/test_path.py` | Локальная проверка PathFinder/SmoothPath и отрисовка `path.png`. |
| `path_orb_experiment/photo_test.py` | Взлет, снимки с OPT-камеры и сохранение ORB-debug изображений. |
| `path_orb_experiment/main.py` | Прямой экспериментальный полет к точкам с ORB-обновлением координат. |

Эти файлы оставлены как отдельный экспериментальный контур. Основной код проекта
использует более модульные версии в `geoscan_mission/trajectory/grid_path.py` и
`geoscan_mission/vision/orb_grid.py`.

## Зависимости

Минимально для локальной отладки нужны:

- Python 3.10+;
- `numpy`;
- `opencv-contrib-python` или другая сборка OpenCV с `cv2.aruco`;
- `scipy` для сглаживания траекторий;
- `tcod` для модульного grid path demo;
- `pioneer_sdk2` на дроне или машине, которая напрямую управляет Pioneer Mini 2.

В репозитории сейчас нет `requirements.txt`, поэтому зависимости ставятся вручную
под конкретную среду запуска.

## Быстрый запуск

Локальная проверка основного пайплайна без взлета:

```bash
python fly_orb_ransac.py --no-flight --reference map.jpg --camera-index 0 --aruco
```

Запуск основного сценария на Mini 2:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --aruco
```

Replay записанного видео без дрона:

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

Калибровка камеры Mini 2:

```bash
python3 calibration.py --camera-source sdk2 --sdk2-camera-type OPT --max-frames 30 --capture-interval 1.0 --output data.yml --frames-dir calibration_frames --debug-dir calibration_debug
```

Пересчет калибровки по уже сохраненным кадрам:

```bash
python calibration.py --images calibration_frames --glob "*.jpg" --output data.yml --debug-dir calibration_debug
```

Проверка одного изображения на ArUco:

```bash
python aruco --image frame.jpg --json --draw frame_aruco.jpg
```

Live hand-check ArUco:

```bash
python aruco_hand_check.py --camera-source sdk2 --sdk2-camera-type OPT --preview-port 8001
```

Миссия поиска и посадки на ID 15:

```bash
python fly_find_id15.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT
```

Готовый профиль полета с записью видео:

```bash
python fly_record_video.py
```

## Работа с траекториями

Основной сценарий поддерживает:

- `--trajectory waypoints` - точки по умолчанию или повторяющиеся `--waypoint x,y,z`;
- `--trajectory square` - квадрат по рабочей зоне;
- `--trajectory lawnmower` - змейка по сетке;
- `--trajectory cube` - несколько слоев lawnmower на разных высотах.

Пример:

```bash
python fly_orb_ransac.py --reference map.jpg --trajectory lawnmower --grid-size 4 --margin 0.25 --height 1.0 --aruco
```

Manual-speed режим:

```bash
python3 fly_orb_ransac.py --reference map.jpg --camera-source sdk2 --sdk2-camera-type OPT --trajectory square --area-size 1.0 --margin 0.2 --height 0.8 --speed 0.12 --move-timeout 20 --control-mode manual-speed --aruco
```

По умолчанию используется `--control-mode autopilot`, где SDK ведет дрон через
`go_to_local_point`. Режим `manual-speed` отправляет частые
`set_manual_speed_body_fixed` команды и требует рабочие
`get_local_position_lps()`/`get_local_yaw_lps()`.

## Работа с картой

В проекте есть рабочая `map.jpg`. Исходный большой TIFF (`Карта_Дроно_старт.tif`)
можно сконвертировать так:

```bash
python tools/convert_map_tif.py Карта_Дроно_старт.tif -o map.jpg --max-side 6000
```

Быстрая проверка локализации по видео или картинке:

```bash
python tools/keypoint_map_localizer.py --reference map.jpg --image frame.jpg --output-dir debug/frame
```

Grid path demo:

```bash
python tools/grid_path_demo.py --map map.jpg --output path_demo.jpg
```

Экспериментальные файлы из `path_orb_experiment/` по умолчанию ждут
`map-2.png`. Если этой карты нет, используйте основной `tools/grid_path_demo.py`
или замените `MAP_FILE` в экспериментальном файле на подходящую карту.

## Логи и артефакты

- `flights/*.csv` - события, локализация и ArUco-логи миссий.
- `flights/*.avi` - clean/overlay видео полетов, если включена запись.
- `debug/` - отладочные изображения локализации.
- `calibration_frames/` - кадры шахматной доски для калибровки.
- `calibration_debug/` - визуальные результаты калибровки.
- `out/` - синтетические/локальные результаты тестов.
- `repomix-output*.md` - локальные справочные дампы, не часть основного кода.

CSV, калибровочные кадры, debug/out и `__pycache__` игнорируются через
`.gitignore`.

## Синхронизация на дрон

PowerShell-скрипт собирает архив проекта, исключая `.git`, `.idea`,
`__pycache__`, `repomix-output*.md` и большие TIFF-файлы, затем загружает его на
`pioneermini@10.42.0.1`:

```powershell
.\sync_to_drone.ps1
```

Перед запуском проверьте IP/пользователя и переменную `$REMOTE_DIR` внутри
`sync_to_drone.ps1`.

## Где что менять

- Полетные команды и safety-логику: `geoscan_mission/flight/control.py`.
- Manual-speed/PID движение: `geoscan_mission/flight/trajectory_control.py`.
- Источники кадров: `geoscan_mission/flight/camera.py`.
- Формы маршрутов: `geoscan_mission/trajectory/patterns.py`.
- Grid path planning: `geoscan_mission/trajectory/grid_path.py`.
- ORB/RANSAC фильтры: `geoscan_mission/vision/localization.py`.
- Grid-ORB детектор: `geoscan_mission/vision/orb_grid.py`.
- ArUco ID, буквы и типы целей: `geoscan_mission/vision/aruco.py` или standalone-версия `aruco_tracker_standalone.py`.
- CSV/JSON/video overlay: `geoscan_mission/recording.py`.
