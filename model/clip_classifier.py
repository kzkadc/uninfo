from typing import override
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import open_clip
from open_clip import CLIP


class CLIPClassifier(nn.Module):
    def __init__(self, clip: CLIP, text_embeddings: Tensor, model_name: str, classes: Sequence[str]):
        super().__init__()

        self.clip = clip
        self.visual = clip.visual

        self.text_embeddings = text_embeddings  # (C,D), normalized
        self.model_name = model_name
        self.classes = classes

    @override
    def forward(self, x: Tensor) -> Tensor:
        z = self.clip.encode_image(x, normalize=True)   # (B,D)
        return self.predict_z(z)

    def predict_z(self, z: Tensor) -> Tensor:
        """
        z is expected to be normalized
        """

        sim = z @ self.text_embeddings.T  # (B,D) @ (D,C) -> (B,C)
        return sim

    def encode_image(self, x: Tensor, normalize: bool = True) -> Tensor:
        return self.clip.encode_image(x, normalize=normalize)


@torch.no_grad()
def create_clip_classifier(arch: str, weight: str | None,
                           class_texts: list[list[str]], classes: Sequence[str]) -> tuple[CLIPClassifier, Callable]:
    clip, _, transform = open_clip.create_model_and_transforms(
        model_name=arch, pretrained=weight, device="cuda")
    clip.eval()

    tokenizer = open_clip.get_tokenizer(arch)
    text_embeddings = []
    for cls_prompts in class_texts:
        cls_tokens = tokenizer(cls_prompts).cuda()
        cls_embedding = clip.encode_text(cls_tokens, normalize=False) \
            .mean(dim=0)
        cls_embedding = F.normalize(cls_embedding, dim=0)
        text_embeddings.append(cls_embedding)

    text_embeddings = torch.stack(text_embeddings)

    classifier = CLIPClassifier(
        clip, text_embeddings, arch, classes)  # type: ignore

    return classifier, transform    # type: ignore
