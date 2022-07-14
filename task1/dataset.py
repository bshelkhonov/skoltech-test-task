from pathlib import Path
import typing as tp

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class EuroSAT(Dataset):
    def __init__(self, directory: str, transforms: tp.Any = None) -> None:
        super().__init__()
        self.files = [str(file) for file in Path(directory).rglob("**/*.jpg")]
        self.cache: dict[int, torch.Tensor] = dict()
        self.transforms = transforms

    def __getitem__(self, index: int) -> torch.Tensor:
        if index in self.cache:
            return self.cache[index]

        file = self.files[index]
        image = read_image(file).float() / 255
        if self.transforms is not None:
            image = self.transforms(image)
        self.cache[index] = image
        return image

    def __len__(self) -> int:
        return len(self.files)
