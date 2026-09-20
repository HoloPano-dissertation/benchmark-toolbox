from __future__ import annotations

import json
from pathlib import Path


def read_room_list(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    values = json.loads(text) if path.suffix == ".json" else text.split()
    rooms = sorted({str(value).strip().strip("/") for value in values})
    if not rooms:
        raise ValueError("Room list is empty: %s" % path)
    return rooms
