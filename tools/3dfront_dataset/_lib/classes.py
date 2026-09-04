DEFAULT_CLASSES = [
    "bed",
    "cabinet_shelf_desk",
    "chair",
    "lighting",
    "other",
    "sofa",
    "stool",
    "table",
]

SOURCE_TO_CLASS = {
    "Bed": "bed",
    "Cabinet_Shelf_Desk": "cabinet_shelf_desk",
    "Chair": "chair",
    "Lighting": "lighting",
    "Others": "other",
    "Pier_Stool": "stool",
    "Sofa": "sofa",
    "Table": "table",
}


def experiment_classes(root):
    path = root / "state" / "classes.json"
    if path.is_file():
        import json
        return list(json.loads(path.read_text(encoding="utf-8"))["classes"])
    return list(DEFAULT_CLASSES)
