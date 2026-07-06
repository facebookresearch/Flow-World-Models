import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class WaymoImageDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.logger = logging.getLogger(__name__)
        self.root = Path(cfg.root)
        self.image_size = int(cfg.image_size)
        self.horizontal_flip = bool(cfg.horizontal_flip)
        self.max_samples = cfg.max_samples
        self.subset_seed = int(cfg.subset_seed)

        self.image_paths = sorted(self.root.glob("*.jpg"), key=self._path_sort_key)

        if self.max_samples is not None:
            max_samples = min(int(self.max_samples), len(self.image_paths))
            generator = torch.Generator().manual_seed(self.subset_seed)
            indices = torch.randperm(len(self.image_paths), generator=generator)[
                :max_samples
            ]
            indices = sorted(indices.tolist())
            self.image_paths = [self.image_paths[idx] for idx in indices]

        self.logger.info(
            "Loaded WaymoImageDataset split=%s from %s with %d images",
            self.split,
            self.root,
            len(self.image_paths),
        )

    @staticmethod
    def _path_sort_key(path: Path):
        return int(path.stem)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)

        image_raw = TF.pil_to_tensor(image)
        backbone_image = image_raw
        image_normalized = image_raw.float().div(127.5).sub(1.0)

        image_id = int(path.stem)
        return {
            "image": image_raw,
            "image_normalized": image_normalized,
            "backbone_image": backbone_image,
            "image_id": image_id,
            "path": str(path),
        }
