from pathlib import Path

import torch
from torch.utils.data import Dataset


class SyntheticBouncingShapesDataset(Dataset):
    def __init__(self, config) -> None:
        self.config = config
        self.root = Path(config.root)
        self._pt_files = sorted(self.root.glob("**/*.pt"))

    def __len__(self) -> int:
        return len(self._pt_files)

    def __getitem__(self, idx: int):
        pt_path = self._pt_files[idx]
        sample = torch.load(pt_path)
        initial_conditions_id, trajectory_id = pt_path.stem.split("_")
        metadata = {
            "initial_conditions_id": initial_conditions_id,
            "trajectory_id": trajectory_id,
        }
        sample["metadata"] = metadata

        return sample
