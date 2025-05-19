import torch
import torch.nn.functional as F
from torch import Tensor


def weighted_entropy(y_pred: Tensor, weight: bool,
                     hard_thresholding: bool, eps: float = 0.5) -> Tensor:
    y_prob = F.softmax(y_pred, dim=1)
    y_log = F.log_softmax(y_pred, dim=1)

    ent = -(y_prob * y_log).sum(dim=1)

    if weight:
        w = torch.exp(eps - ent).detach()
        ent *= w
    if hard_thresholding:
        mask = (ent < eps).float().detach()
        ent *= mask

    ent = ent.mean()
    return ent


def entropy(y_pred: Tensor) -> Tensor:
    return weighted_entropy(y_pred, weight=False, hard_thresholding=False)


def uniformity_loss(z: Tensor, var: float = 1.0) -> Tensor:
    """
    z: (B,D)
    """

    zz = z @ z.T   # (B,B)
    rz = zz.diag().unsqueeze(0).expand_as(zz)
    dist = rz + rz.T - zz * 2

    loss = torch.exp(-0.5 * dist / var).mean().log()
    return loss


def mutual_info(y_pred: Tensor) -> Tensor:
    y_prob = F.softmax(y_pred, dim=1)
    y_log_prob = F.log_softmax(y_pred, dim=1)
    conditional_ent = -(y_prob * y_log_prob).sum(dim=1).mean()

    marginal_prob = y_prob.mean(dim=0)
    marginal_ent = -(marginal_prob * torch.log(marginal_prob)).sum()

    return marginal_ent - conditional_ent
