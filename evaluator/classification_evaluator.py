from dataclasses import dataclass, InitVar

import torch
import torch.nn.functional as F
from torch import Tensor
import ignite.engine
from ignite.metrics import Accuracy, Loss, Entropy, MutualInformation, Metric, Average, GpuInfo
from ignite.handlers import EMAHandler

from model.clip_classifier import CLIPClassifier
from util.engine import Engine
from util.loss import uniformity_loss


@dataclass
class CLIPAdaptationEvaluator(Engine):
    net: CLIPClassifier
    tau: float
    ema_handler: InitVar[EMAHandler | None]
    unif_loss_config: InitVar[dict]

    def __post_init__(self, ema_handler: EMAHandler | None, unif_loss_config: dict):
        super().__init__(self.inference)

        gpu_info = GpuInfo()
        ot = lambda d: (d["y_pred"], d["y"])
        self.metrics: dict[str, Metric] = {
            "accuracy": Accuracy(ot),
            "cross_entropy": Loss(F.cross_entropy, ot),
            "entropy": Entropy(ot),
            "infomax": MutualInformation(ot),
            "unif_loss": Average(lambda d: uniformity_loss(d["z"], **unif_loss_config)),
            "gpu_usage": Average(lambda _: gpu_info.compute()[0]["fb_memory_usage"]["used"])
        }
        for name, metric in self.metrics.items():
            metric.attach(self, name)

        if ema_handler is not None:
            print("Inference with EMA model")
            self.net = ema_handler.ema_model   # type: ignore

    @torch.no_grad()
    def inference(self, engine: ignite.engine.Engine, batch: tuple[Tensor, Tensor]) -> dict:
        self.net.eval()

        x, y = batch
        x = x.cuda()
        y = y.cuda()

        z = self.net.encode_image(x, normalize=True)
        y_pred = self.net.predict_z(z) / self.tau

        return {
            "y_pred": y_pred,
            "y": y,
            "z": z
        }
