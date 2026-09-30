from __future__ import annotations

import numpy as np


def manhattan_world(layout):
    ring = np.asarray(layout["polygon"]["coordinates"][0], dtype=float)[:-1, :2]
    floor = np.column_stack((ring, np.full(len(ring), float(layout["floor_z"]))))
    ceiling = np.column_stack((ring, np.full(len(ring), float(layout["ceiling_z"]))))
    corners = np.empty((2 * len(ring), 3), dtype=float)
    corners[0::2] = ceiling
    corners[1::2] = floor
    return corners


def share_one_column(corners_pix):
    corners_pix = np.array(corners_pix, dtype=float)
    corners_pix[1::2, 0] = corners_pix[0::2, 0]
    return corners_pix


def sorted_by_column(corners_pix, corners_world):
    corners_pix = share_one_column(corners_pix)
    corners_world = np.asarray(corners_world, dtype=float)
    order = np.argsort(corners_pix[0::2, 0], kind="stable")
    index = np.empty(len(corners_pix), dtype=int)
    index[0::2] = order * 2
    index[1::2] = order * 2 + 1
    return corners_pix[index], corners_world[index]
