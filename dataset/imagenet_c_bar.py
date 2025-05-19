from typing import Any
from collections.abc import Callable
import tarfile
import json
import io
from pathlib import Path

from PIL import Image

from dataset.dataset_config import IMAGENET_C_BAR_ROOT
from dataset.dataset_utils import VisionClassificationDataset


with open("dataset/wnid_to_idx.json", "r", encoding="utf-8") as f:
    WNID_TO_IDX: dict[str, int] = json.load(f)

with open("dataset/idx_to_classes.json", "r", encoding="utf-8") as f:
    IDX_TO_CLASSES: dict[int, list[str]] = {
        int(i): c
        for i, c in json.load(f).items()
    }


class ImageNetCBar(VisionClassificationDataset):
    def __init__(self, corruption: str, severity: int,
                 transform: Callable = lambda x: x):
        super().__init__(root=IMAGENET_C_BAR_ROOT, transform=transform)

        tar_path = Path(self.root, f"{corruption}.tar")
        self.tar = tarfile.open(str(tar_path), "r")

        self.members = tuple(
            m
            for m in self.tar.getmembers()
            if m.isfile() and m.name.startswith(f"{corruption}/{severity}/")
        )
        assert len(self.members) > 0, \
            f"Invalid corruption or severity: {corruption!r}, {severity}"

        self.wnids = tuple(
            m.name.split("/")[-2]
            for m in self.members
        )
        self.class_indices = tuple(
            WNID_TO_IDX[wnid]
            for wnid in self.wnids
        )
        self.classes = tuple(
            ", ".join(IDX_TO_CLASSES[i])
            for i in sorted(set(self.class_indices))
        )
        self.labels = self.class_indices

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        fp = self.tar.extractfile(self.members[i])
        assert fp is not None

        with io.BytesIO(fp.read()) as bio:
            img = Image.open(bio).convert("RGB")
        img = self.transform(img)   # type: ignore

        return img, self.class_indices[i]

    def close(self):
        self.tar.close()
