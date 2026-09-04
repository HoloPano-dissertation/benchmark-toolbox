# Обучение компонент на 3D-FRONT

Шесть обучений. Детектор и оценщик планировки не зависят ни от масштаба, ни от
классов и запускаются первыми; остальные читают сцены и каталог объектов.

| Порядок | Компонента | Конфигурация | Читает |
| ---: | --- | --- | --- |
| 1 | Детектор | `tools/3dfront_training/train_detector.py` | `coco/` |
| 2 | HorizonNet | `tools/3dfront_training/train_horizonnet.py` | `horizonnet/` |
| 3 | BEN | `bdb3d_estimation_front3d.yaml` | `dpc_scenes/` |
| 4 | LDIF | `ldif_front3d.yaml` | `objects/` |
| 5 | MGN | `mgnet_front3d.yaml` | `objects/` |
| 6 | Scene-GCN | `relation_scene_gcn_front3d.yaml` | `dpc_scenes/`, `objects/` |

Шаг 6 требует готовых весов шагов 3 и 4: подставьте их пути в `weight`.
Из шести компонент методы собираются так: DPC — все шесть; Im3D-Pano — без
Scene-GCN; Total3D-Pano — без Scene-GCN и с MGN вместо LDIF.

## Перед запуском

Замените `/PATH/TO/front3d` на папку эксперимента: конфигурации DPC не раскрывают
переменные окружения в путях.

Список классов задаётся переменной окружения, иначе окружение подставит словарь
iGibson из 56 категорий:

```bash
export PANO3D_CLASSES=$(python -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['classes']))" \
  /PATH/TO/front3d/state/classes.json)
```

Переменную читает патч из `configs/environments/dpc.yaml`, который применяется при
развёртывании окружения. Без переменной поведение прежнее, воспроизведение на
iGibson не ломается.

BEN обучается с нуля: у предобученных весов размерность классового кода равна 56,
и к нашему списку они не подходят.
