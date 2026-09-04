from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DenseLayoutDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = Path(root)
        self.images = sorted((self.root / "img").glob("*.png"))
        if not self.images:
            raise ValueError("Empty HorizonNet split")
        self.augment = augment
        for path in self.images:
            if not (self.root / "label_dense" / (path.stem+".npz")).is_file():
                raise FileNotFoundError(path.stem)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        with np.load(self.root / "label_dense" / (path.stem+".npz"), allow_pickle=False) as target:
            boundary, corner = target["boundary"], target["corner"]
        if self.augment:
            if np.random.randint(2):
                image, boundary, corner = image[:, ::-1], boundary[:, ::-1], corner[:, ::-1]
            shift = np.random.randint(image.shape[1])
            image = np.roll(image, shift, axis=1)
            boundary, corner = np.roll(boundary, shift, axis=1), np.roll(corner, shift, axis=1)
            gamma = np.random.uniform(1, 2)
            image = image**(gamma if np.random.randint(2) else 1/gamma)
        return [torch.from_numpy(image.transpose(2, 0, 1).copy()),
                torch.from_numpy(boundary.copy()), torch.from_numpy(corner.copy())]
