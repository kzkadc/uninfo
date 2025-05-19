from typing import Any
from collections.abc import Callable, Sequence

from torchvision.datasets import VisionDataset


class VisionClassificationDataset(VisionDataset):
    classes: Sequence[str]
    labels: Sequence[int]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[Any, int]:
        raise NotImplementedError

    def close(self):
        pass


class TransformDataset(VisionClassificationDataset):
    def __init__(self, dataset: VisionClassificationDataset, transform: Callable):
        super().__init__(dataset.root, transform=transform)

        self.dataset = dataset

        self.labels = dataset.labels
        self.classes = dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        img, label = self.dataset[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class VisionClassificationSubset(VisionClassificationDataset):
    def __init__(self, dataset: VisionClassificationDataset, indices: Sequence[int]):
        super().__init__(dataset.root)
        self.dataset = dataset
        self.indices = indices

        self.labels = tuple(dataset.labels[i] for i in indices)
        self.classes = dataset.classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        return self.dataset[self.indices[i]]
