"""Angle encoding and decoding utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn


class UCResolver:
    """Unit Cycle Resolver for periodic angle representations."""

    def __init__(
        self,
        angle_version: str,
        mdim: int = 3,
        invalid_thr: float = 0.0,
        loss_angle_restrict: Callable[[Tensor, Tensor], Tensor] | nn.Module | dict[str, Any] | None = None,
    ) -> None:
        """Initialize resolver with encoding dimension and optional restriction loss."""

        if angle_version not in {"le90"}:
            raise AssertionError("Only 'le90' angle_version is supported.")
        if mdim < 2:
            raise AssertionError("'mdim' must be greater than or equal to 2.")

        self.angle_version = angle_version
        self.mdim = mdim
        self.invalid_thr = invalid_thr
        self.encode_size = mdim

        if loss_angle_restrict is None:
            self.loss_angle_restrict: Callable[[Tensor, Tensor], Tensor] | None = None
        else:
            self.loss_angle_restrict = self._build_loss(loss_angle_restrict)

        base_angles = torch.arange(self.mdim, dtype=torch.float32) * (2 * torch.pi / self.mdim)
        self.coef_sin = torch.sin(base_angles)
        self.coef_cos = torch.cos(base_angles)

    @staticmethod
    def _build_loss(
        cfg: Callable[[Tensor, Tensor], Tensor] | nn.Module | dict[str, Any]
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a loss function from callable, module or config dict."""

        if callable(cfg):
            return cfg
        if isinstance(cfg, nn.Module):
            return cfg
        if isinstance(cfg, dict):
            cfg = cfg.copy()
            loss_type = cfg.pop("type", None)
            if loss_type is None or not hasattr(nn, loss_type):
                raise ValueError("Invalid loss config for UCResolver.")
            loss_cls = getattr(nn, loss_type)
            if not issubclass(loss_cls, nn.Module):
                raise ValueError("Loss type must be a torch.nn.Module subclass.")
            return loss_cls(**cfg)
        raise TypeError("Unsupported loss specification for UCResolver.")

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Encode target angles into unit-cycle representation."""

        angle_targets = angle_targets * 2
        if self.mdim > 2:
            phase = angle_targets.unsqueeze(-1) + (2 * torch.pi / self.mdim) * torch.arange(
                self.mdim, device=angle_targets.device, dtype=angle_targets.dtype
            )
            encoded_targets = torch.cos(phase).squeeze(-2)
        else:
            encoded_targets = torch.cat([torch.cos(angle_targets), torch.sin(angle_targets)], dim=-1)
        return encoded_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Decode encoded predictions back to angle values."""

        coef_sin = self.coef_sin.to(angle_preds)
        coef_cos = self.coef_cos.to(angle_preds)

        if self.mdim > 2:
            predict_cos = torch.sum(angle_preds * coef_cos, dim=-1, keepdim=keepdim)
            predict_sin = -torch.sum(angle_preds * coef_sin, dim=-1, keepdim=keepdim)
        else:
            predict_cos = angle_preds[..., 0, None]
            predict_sin = angle_preds[..., 1, None]

        theta = torch.atan2(predict_sin, predict_cos)

        if self.invalid_thr > 0:
            mask = predict_sin.square() + predict_cos.square() < (self.mdim / 2) ** 2 * self.invalid_thr
            theta = theta.clone()
            theta[mask] = 0

        return theta / 2

    def get_restrict_loss(self, angle_preds: Tensor) -> Tensor:
        """Compute restriction loss to keep predictions on the unit cycle."""

        if self.loss_angle_restrict is None:
            raise RuntimeError("loss_angle_restrict is not defined for UCResolver.")

        d_angle_restrict = torch.sum(angle_preds.pow(2), dim=-1)
        d_angle_target = torch.ones_like(d_angle_restrict) * (self.mdim / 2)
        loss = self.loss_angle_restrict(d_angle_restrict, d_angle_target)

        if self.mdim == 3:
            d_angle_restrict = torch.sum(angle_preds, dim=-1)
            d_angle_target = torch.zeros_like(d_angle_restrict)
            loss = loss + self.loss_angle_restrict(d_angle_restrict, d_angle_target)

        return loss

