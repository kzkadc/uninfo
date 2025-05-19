from typing import Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
import ignite.engine
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Entropy, MutualInformation, Metric, Average, MetricsLambda, GpuInfo
from ignite.handlers import EMAHandler

from model.clip_classifier import CLIPClassifier

from util.engine import Engine
from util.loss import uniformity_loss, weighted_entropy, mutual_info


@dataclass
class CLIPTTAEngine(Engine):
    net: CLIPClassifier
    opt: torch.optim.Optimizer
    ent_config: dict
    unif_loss_config: dict
    ema_handler: EMAHandler | None
    ema_reg_lam: float
    lam_unif: float
    info_weight_th: float
    tau_s: float
    tau_t: float

    def __post_init__(self):
        super().__init__(self.update)

        gpu_info = GpuInfo()
        y_ot = lambda d: (d["y_pred"], d["y"])
        self.metrics: dict[str, Metric] = {
            "accuracy": Accuracy(y_ot),
            "entropy": Entropy(y_ot),
            "cross_entropy": Loss(F.cross_entropy, y_ot),
            "infomax": MutualInformation(y_ot),
            "unif_loss": MetricsLambda(
                lambda avg: avg.item(),
                Average(lambda d: uniformity_loss(
                    d["z"], **self.unif_loss_config))
            ),
            "gpu_usage": MetricsLambda(
                lambda avg: avg.item(),
                Average(lambda _: gpu_info.compute()[
                        0]["fb_memory_usage"]["used"])
            )
        }

        if self.ema_handler is not None:
            print("EMA is enabled")
            self.ema_handler.attach(self, event=Events.ITERATION_COMPLETED)

            y_ot_ema = lambda d: (d["y_pred_ema"], d["y"])
            soft_y_ot = lambda d: (
                d["y_pred"], F.softmax(d["y_pred_ema"], dim=1))
            self.metrics.update({
                "accuracy_ema": Accuracy(y_ot_ema),
                "entropy_ema": Entropy(y_ot_ema),
                "cross_entropy_ema": Loss(F.cross_entropy, y_ot_ema),
                "infomax_ema": MutualInformation(y_ot_ema),
                "teacher-student_ce": Loss(F.cross_entropy, soft_y_ot),
                "unif_loss_ema": MetricsLambda(
                    lambda avg: avg.item(),
                    Average(lambda d: uniformity_loss(
                        d["z_ema"], **self.unif_loss_config))
                )
            })

        for name, metric in self.metrics.items():
            metric.attach(self, name)

    def update(self, engine: ignite.engine.Engine, batch: tuple[Tensor, Tensor]) -> dict:
        self.net.train()
        self.net.zero_grad()

        x, y = batch
        x = x.cuda()
        y = y.cuda()

        z = self.net.encode_image(x, normalize=True)
        y_pred: Tensor = self.net.predict_z(z) / self.tau_s

        output: dict[str, Any] = {
            "y_pred": y_pred,
            "y": y,
            "z": z
        }

        ent_loss = weighted_entropy(y_pred, **self.ent_config)
        unif_loss = uniformity_loss(z, **self.unif_loss_config)

        mi = mutual_info(y_pred)
        mi_weight = torch.exp(mi - self.info_weight_th).detach()
        ent_loss *= mi_weight
        unif_loss /= mi_weight

        loss = ent_loss + self.lam_unif * unif_loss

        if self.ema_handler is not None:
            ema_clip: CLIPClassifier = self.ema_handler.ema_model   # type: ignore
            ema_clip.eval()
            with torch.no_grad():
                z_ema = ema_clip.encode_image(x, normalize=True)
                y_pred_ema = ema_clip.predict_z(z_ema)
            ema_clip.train()

            y_pred_ema /= self.tau_t

            pl = F.softmax(y_pred_ema, dim=1)

            loss += self.ema_reg_lam * F.cross_entropy(y_pred, pl)

            output.update({
                "y_pred_ema": y_pred_ema,
                "z_ema": z_ema
            })

        loss.backward()
        self.opt.step()

        return output
