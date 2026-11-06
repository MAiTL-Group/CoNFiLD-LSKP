"""Core models for the Predictive Koopman Chart Prior (PKCP)."""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn



def _normalize_hidden_dims(hidden_dims: Optional[Iterable[int]]) -> List[int]:
    if hidden_dims is None:
        return []
    return [int(x) for x in hidden_dims]



def build_mlp(in_dim: int, out_dim: int, hidden_dims: Optional[Iterable[int]]) -> nn.Sequential:
    dims = [int(in_dim)] + _normalize_hidden_dims(hidden_dims) + [int(out_dim)]
    layers = []
    for idx in range(len(dims) - 2):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        layers.append(nn.GELU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PredictiveDynEncoder(nn.Module):
    """Map aligned CNF latents to a predictive Koopman subspace."""

    def __init__(
        self,
        dim_input: int,
        dim_dyn: int,
        hidden_dims: Optional[Iterable[int]] = None,
        context_length: int = 1,
        temporal_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim_input = int(dim_input)
        self.dim_dyn = int(dim_dyn)
        self.context_length = int(context_length)
        if self.context_length <= 0:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")
        if self.context_length == 1:
            self.temporal = None
            self.head = build_mlp(self.dim_input, self.dim_dyn, hidden_dims)
            self.net = self.head
        else:
            temporal_hidden = int(temporal_hidden_dim or max(self.dim_input, self.dim_dyn))
            self.temporal = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.dim_input,
                    out_channels=temporal_hidden,
                    kernel_size=self.context_length,
                    bias=True,
                ),
                nn.GELU(),
            )
            self.head = build_mlp(temporal_hidden, self.dim_dyn, hidden_dims)
            self.net = self.head

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        if self.context_length == 1:
            if u_t.ndim == 3:
                if u_t.shape[1] != 1:
                    raise ValueError(
                        f"context_length=1 expects [B,D] or [B,1,D], got {tuple(u_t.shape)}"
                    )
                u_t = u_t[:, 0, :]
            if u_t.ndim != 2:
                raise ValueError(f"context_length=1 expects [B,D], got {tuple(u_t.shape)}")
            return self.head(u_t)
        if u_t.ndim != 3:
            raise ValueError(
                f"context_length={self.context_length} expects [B,L,D], got {tuple(u_t.shape)}"
            )
        if u_t.shape[1] != self.context_length or u_t.shape[2] != self.dim_input:
            raise ValueError(
                f"context input mismatch: expected [B,{self.context_length},{self.dim_input}], "
                f"got {tuple(u_t.shape)}"
            )
        temporal = self.temporal(u_t.transpose(1, 2)).squeeze(-1)
        return self.head(temporal)


class CoarseDynHead(nn.Module):
    """Map the dynamical code to a fixed coarse observable target."""

    def __init__(
        self,
        dim_dyn: int,
        dim_coarse: int,
        hidden_dims: Optional[Iterable[int]] = None,
        linear_only: bool = False,
    ):
        super().__init__()
        self.dim_dyn = int(dim_dyn)
        self.dim_coarse = int(dim_coarse)
        self.linear_only = bool(linear_only)
        if self.linear_only:
            self.net = nn.Linear(self.dim_dyn, self.dim_coarse, bias=False)
        else:
            self.net = build_mlp(self.dim_dyn, self.dim_coarse, hidden_dims)

    def forward(self, z_dyn: torch.Tensor) -> torch.Tensor:
        return self.net(z_dyn)


class AuxEncoder(nn.Module):
    """Residual/detail encoder conditioned on the frozen dynamical code."""

    def __init__(self, dim_input: int, dim_dyn: int, dim_aux: int, hidden_dims: Optional[Iterable[int]] = None):
        super().__init__()
        self.dim_input = int(dim_input)
        self.dim_dyn = int(dim_dyn)
        self.dim_aux = int(dim_aux)
        self.net = build_mlp(self.dim_input + self.dim_dyn, self.dim_aux, hidden_dims)

    def forward(self, u_t: torch.Tensor, z_dyn: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u_t, z_dyn], dim=-1))


class ChartDecoder(nn.Module):
    """Decode a predictive dyn code plus residual code back to aligned CNF latent space.

    The decoder is decomposed as:
        g(z_dyn, z_aux) = g_coarse(z_dyn) + g_res(z_dyn, z_aux)
    to force the dyn code to explain coarse structure.
    """

    def __init__(
        self,
        dim_input: int,
        dim_dyn: int,
        dim_aux: int,
        hidden_dims_coarse: Optional[Iterable[int]] = None,
        hidden_dims_res: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        self.dim_input = int(dim_input)
        self.dim_dyn = int(dim_dyn)
        self.dim_aux = int(dim_aux)
        self.coarse = build_mlp(self.dim_dyn, self.dim_input, hidden_dims_coarse)
        self.residual = build_mlp(self.dim_dyn + self.dim_aux, self.dim_input, hidden_dims_res)

    def forward(self, z_dyn: torch.Tensor, z_aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse = self.coarse(z_dyn)
        residual = self.residual(torch.cat([z_dyn, z_aux], dim=-1))
        return coarse + residual, coarse, residual
