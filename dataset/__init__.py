from collections.abc import Callable

from .dataset_utils import VisionClassificationDataset
from .imagenet_c import ImageNetC
from .imagenet_c_bar import ImageNetCBar


def get_val_dataset(config: dict, transform: Callable = lambda x: x) -> VisionClassificationDataset:
    match config["dataset"]["name"]:
        case "imagenet-c":
            ds = ImageNetC(transform=transform, **config["dataset"]["config"])
        case "imagenet-c-bar":
            ds = ImageNetCBar(transform=transform,
                              **config["dataset"]["config"])

        case x:
            raise ValueError(f"Invalid dataset name: {x!r}")

    return ds
