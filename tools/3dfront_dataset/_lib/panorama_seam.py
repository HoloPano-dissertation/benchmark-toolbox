import numpy as np

FRAMES = (0, 1)


def frame_shift(width, frame):
    if frame not in FRAMES:
        raise ValueError(f"Unknown panorama frame: {frame}")
    return 0 if frame == 0 else int(width) // 2


def to_frame(array, frame):
    array = np.asarray(array)
    shift = frame_shift(array.shape[1], frame)
    return array if shift == 0 else np.roll(array, shift, axis=1)


def touches_both_edges(mask):
    mask = np.asarray(mask)
    columns = mask.any(axis=0)
    return bool(columns[0] and columns[-1])


def widest_empty_gap(mask):
    columns = np.asarray(mask).any(axis=0)
    empty = np.flatnonzero(~columns)
    if not len(empty):
        return None
    breaks = np.flatnonzero(np.diff(empty) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(empty) - 1]))
    lengths = empty[ends] - empty[starts] + 1
    best = int(np.argmax(lengths))
    return int(empty[starts[best]]), int(lengths[best])


def split_at_widest_gap(mask):
    mask = np.asarray(mask)
    gap = widest_empty_gap(mask)
    if gap is None:
        return [mask]
    start, length = gap
    cut = start + length // 2
    left, right = np.zeros_like(mask), np.zeros_like(mask)
    left[:, :cut] = mask[:, :cut]
    right[:, cut:] = mask[:, cut:]
    return [piece for piece in (left, right) if piece.any()]


def annotation_masks(mask):
    mask = np.asarray(mask)
    if not touches_both_edges(mask):
        return [(0, mask)]
    rolled = to_frame(mask, 1)
    if not touches_both_edges(rolled):
        return [(1, rolled)]
    return [(0, piece) for piece in split_at_widest_gap(mask)]


def tight_box(mask):
    rows, columns = np.nonzero(np.asarray(mask))
    if not len(columns):
        return None
    x_min, x_max = int(columns.min()), int(columns.max())
    y_min, y_max = int(rows.min()), int(rows.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
