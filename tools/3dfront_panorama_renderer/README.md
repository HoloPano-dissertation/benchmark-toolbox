# 3D-FRONT panorama renderer

Самостоятельный рендерер комнат из **переработанного MIDI-3D/3D-FRONT**:
[huanngzh/3D-Front](https://huggingface.co/datasets/huanngzh/3D-Front).
Не читает raw JSON 3D-FRONT.

## Вход и выход

Вход — одна распакованная комната `HOUSE_ID/ROOM_ID/` с отдельными
`*.glb` для конструкций и предметов (например, `floor.glb`, `wall.glb`,
`ceil.glb`, `Bed_<uuid>_0.glb`). Сохраняйте исходные имена и структуру.

Выход по умолчанию — четыре эквиректангулярные панорамы 1024×512:

- `0.hdf5` … `3.hdf5`: RGB `colors`, `depth`, `normals`,
  `instance_segmaps` и `instance_attribute_maps` с исходными именами/классами.
- `render.json`: камеры, восстановленный контур пола, параметры, масштаб
  отсечения камеры и хеши реализации для воспроизводимости.
- `.complete` создаётся **пакетным запуском** после проверки наличия всех кадров.

Depth может содержать невалидные значения в направлениях без поверхности;
это не готовая плотная depth-разметка. Единицы координат — единицы исходного GLB,
их соответствие метрам отдельно не подтверждено.

## Установка и одна комната

Проверено с BlenderProc 2.8.0 и Blender 4.2.1. Команды из корня репозитория:

```bash
python3 -m venv .venv-render
source .venv-render/bin/activate
pip install -r tools/3dfront_panorama_renderer/requirements.txt
blenderproc pip install "shapely>=2,<3"

bash tools/3dfront_panorama_renderer/run_room.sh \
  /data/3D-FRONT-TEST-SCENE/HOUSE_ID/ROOM_ID /data/rendered-room \
  --views 4 --width 1024 --height 512 --samples 32 --min-clearance 0.1

python tools/3dfront_panorama_renderer/extract_preview.py \
  /data/rendered-room/0.hdf5 /data/rendered-room/preview
```

При установленном Blender можно задать `BLENDER_INSTALL_PATH`;
при нестандартном окружении — `BLENDERPROC_BIN`.
BlenderProc запускает Blender без GUI; Linux всё равно требует его системные
библиотеки. Для первого запуска BlenderProc может скачать Blender.

Камеры выбираются над реальным полом, внутри его контура и **вне основания
мебели**: положение отбраковывается, если камера оказывается над предметом,
поднимающимся выше десятой доли высоты комнаты. Проверки одного лишь трёхмерного
габарита недостаточно — камера над кроватью выше её коробки, а ближайшая
поверхность под ней далека, поэтому такое положение проходило и зазор, и габарит.
Ковёр или порожек проходу не мешают. Восстановленный контур обрезается по границам оболочки комнаты
(`<дом>/<комната>.glb`): лежащий рядом `floor.glb` часто несёт плиту пола всего
жилья, и без обрезки планировка вместе с областью установки камер выходит за
пределы комнаты. Пол, потолок и стены не достраиваются. При невозможности найти безопасные
позиции рендер завершается ошибкой. `--plan-only` проверяет позиции и пишет
метаданные без HDF5. Для проверки используйте отдельную выходную папку.

## Несколько комнат

JSONL: одна запись на комнату, например:

```json
{"room_id":"house/room","split":"train","room_dir":"/data/house/room","min_clearance":0.1}
```

```bash
python tools/3dfront_panorama_renderer/run_batch.py /data/rooms.jsonl /data/outputs
```

Результат: `outputs/SPLIT/HOUSE_ID/ROOM_ID/`.
Для нескольких независимых процессов есть `--shard-count N --shard-index I`
(`I` от 0 до `N-1`). Slurm не требуется.

Возобновление пропускает готовую комнату только при совпадении кода и параметров.
После изменения геометрии источника используйте новую выходную папку:
хеши кода не проверяют содержимое исходных GLB.

## Что лежит в этой папке

`render.py` — Blender; `camera_policy.py` — камеры;
`room_layout.py` и `glb_geometry.py` — геометрия;
`run_room.sh` и `run_batch.py` — запуск;
`extract_preview.py` — просмотр HDF5; `requirements.txt` — зависимости.

Разметка, фиксированные split и контроль качества — в
[3dfront_dataset](../3dfront_dataset/README.md).
Обучение — в [3dfront_training](../3dfront_training/README.md).
Данные, журналы запусков и отчёты не входят в рендерер.
