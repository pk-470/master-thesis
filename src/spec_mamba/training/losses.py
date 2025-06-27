"""Custom loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedLoss(nn.Module):
    """
    Masked loss. Calculates the chosen loss only on masked elements,
    as indicated by the mask tensor passed in the forward method.
    Input: x, target, mask.
    """

    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.loss_fn(input * mask, target * mask)
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss function.
    Input: a Tensor of shape (2 * batch_size, feature_dim),
    where each positive pair is offset by batch_size.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[0] % 2 == 0

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, float("-inf"))

        # Find positive example batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        cos_sim /= self.temperature
        loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ContRecLoss(nn.Module):
    """
    Contrastive reconstruction loss.
    Calculates a weighted sum of the chosen reconstruction and contrastive losses.
    Input: cont, rec, target, mask.
    """

    def __init__(
        self,
        rec_loss: nn.Module = nn.MSELoss(),
        cont_loss: nn.Module = InfoNCELoss(),
        cont_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.rec_loss = rec_loss
        self.cont_loss = cont_loss
        self.cont_loss_weight = cont_loss_weight

    def forward(
        self, cont: Tensor, rec: Tensor, target: Tensor, mask: Tensor
    ) -> dict[str, Tensor]:
        rec_loss = (
            self.rec_loss(rec, target, mask)
            if isinstance(self.rec_loss, MaskedLoss)
            else self.rec_loss(rec, target)
        )
        cont_loss = self.cont_loss(cont)
        loss = rec_loss + self.cont_loss_weight * cont_loss

        return {"loss": loss, "rec_loss": rec_loss, "cont_loss": cont_loss}
