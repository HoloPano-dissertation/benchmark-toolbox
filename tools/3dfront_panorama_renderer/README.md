# Рендер панорам из MIDI-3D/3D-FRONT

Самостоятельный рендерер обработанных комнат из публичного набора
[MIDI-3D/3D-FRONT](https://huggingface.co/datasets/huanngzh/3D-Front). Он принимает
папку комнаты с отдельными `.glb`-файлами и создаёт эквиректангулярные RGB-панорамы
вместе с depth, normals и instance masks.

Рендерер рассчитан на **обработанный MIDI-формат с GLB-комнатами**. Исходные JSON
raw 3D-FRONT напрямую он не читает.

## Вход

Одна папка комнаты:

```text
HOUSE_UUID/Bedroom-1234/
├── Bed_UUID_1.glb
├── Cabinet_UUID_2.glb
├── floor.glb
├── wall.glb
└── ceil.glb
```

Все `.glb` импортируются в одну сцену. Метка объекта берётся из имени файла;
исходное имя также сохраняется в instance attributes.

## Установка

Нужны Python 3.9+, BlenderProc и Blender. Проверенная связка на кластере:
BlenderProc 2.8.0 + Blender 4.2.1.

```bash
python3 -m venv .venv-front3d-render
source .venv-front3d-render/bin/activate
pip install -r tools/3dfront_panorama_renderer/requirements.txt
```

BlenderProc может скачать Blender при первом запуске. Если Blender уже установлен
в подготовленную BlenderProc-папку, задайте её через `BLENDER_INSTALL_PATH`.

## Одна комната

Из корня репозитория:

```bash
tools/3dfront_panorama_renderer/run_room.sh \
  /data/3D-FRONT-SCENE/HOUSE_UUID/Bedroom-1234 \
  /data/panoramas/HOUSE_UUID/Bedroom-1234 \
  --views 4 --width 1024 --height 512 --samples 32
```

Полезные переменные окружения:

- `BLENDERPROC_BIN` — путь к `blenderproc`, если его нет в `PATH`;
- `BLENDER_INSTALL_PATH` — подготовленная папка BlenderProc с Blender;
- `BLENDERPROC_TEMP_DIR` — собственная временная папка; иначе используется `mktemp`.

Параметры `render.py`:

- `--views` — число точек камеры, по умолчанию 4;
- `--width`, `--height` — разрешение, по умолчанию 1024×512;
- `--samples` — Cycles samples, по умолчанию 32;
- `--camera-height` — высота камеры в единицах GLB;
- `--min-clearance` — минимальное расстояние камеры до ближайшей геометрии.

## Выход

```text
OUTPUT_DIR/
├── 0.hdf5
├── 1.hdf5
├── 2.hdf5
├── 3.hdf5
└── render.json
```

Каждый HDF5 соответствует одной точке камеры и содержит:

- `colors` — RGB, `uint8`, `[H, W, 3]`;
- `depth` — depth, `float32`, `[H, W]`;
- `normals` — нормали, `float32`, `[H, W, 3]`;
- `instance_segmaps` — instance ID каждого пикселя;
- `instance_attribute_maps` — соответствие ID имени меша, исходному GLB и метке.

`render.json` содержит пути, границы комнаты, координаты камер, разрешение и параметры
рендера. Значения depth выражены в единицах входного GLB: считать их метрами можно
только после отдельной проверки масштаба набора.

Чтобы посмотреть результат без HDF5-viewer:

```bash
python tools/3dfront_panorama_renderer/extract_preview.py \
  /data/panoramas/HOUSE_UUID/Bedroom-1234/0.hdf5 \
  /tmp/front3d-preview
```

Будут созданы `colors.png`, `depth.png`, `normals.png`, `instances.png` и описание
datasets. `depth.png` — только визуализация; исходные float32 значения остаются в
HDF5.

## Ограничения

- Камера выбирается автоматически по bounding box и расстоянию до геометрии. Для
  узких или необычных помещений отдельные точки могут оказаться близко к стене;
  перед большим обучением полезно визуально проверить случайную выборку.
- Скрипт создаёт синтетическое освещение, а не воспроизводит исходный MIDI render.
