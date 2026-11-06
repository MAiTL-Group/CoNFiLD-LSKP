import torch
import torch.nn as nn
import torch.nn.functional as F


class SNLinear(nn.Module):
    """
    Spectral-normalized linear layer with configurable power iterations,
    scaling and orthogonal initialization.
    """

    def __init__(self, in_dim, out_dim, bias=True, coeff=1.0, n_power_iter=1, orthogonal_init=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.coeff = 1.0 if coeff is None else coeff
        self.n_power_iter = 1 if n_power_iter is None else int(n_power_iter)
        self.orthogonal_init = orthogonal_init
        # When False, forward uses frozen buffer values (no in-place PI updates).
        self.buffer_update_enabled = True
        self.register_buffer("u", torch.randn(1, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.orthogonal_init:
            nn.init.orthogonal_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)
        else:
            self.fc.reset_parameters()
        with torch.no_grad():
            self.u = F.normalize(self.u, dim=1)

    def _sn_weight(self):
        W = self.fc.weight
        u = self.u
        n_iter = max(1, self.n_power_iter)
        v = F.normalize(u @ W, dim=1, eps=1e-12)
        for _ in range(n_iter):
            v = F.normalize(u @ W, dim=1, eps=1e-12)
            u = F.normalize(v @ W.t(), dim=1, eps=1e-12)
        if self.buffer_update_enabled:
            self.u = u.detach()
        # Static mode uses power-iteration results without updating buffers.
        sigma = torch.sum(u @ W @ v.t())
        W_bar = self.coeff * W / (sigma + 1e-12)
        return W_bar, self.fc.bias

    def forward(self, x):
        W_bar, b = self._sn_weight()
        return F.linear(x, W_bar, b)


def make_mlp(in_dim, out_dim, hidden_dims, coeff=1.0, n_power_iter=1, orthogonal_init=True):
    """
    Build an MLP from SNLinear and GELU layers.
    """
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(
            SNLinear(
                dims[i],
                dims[i + 1],
                coeff=coeff,
                n_power_iter=n_power_iter,
                orthogonal_init=orthogonal_init,
            )
        )
        layers.append(nn.GELU())
    layers.append(
        SNLinear(
            dims[-2],
            dims[-1],
            coeff=coeff,
            n_power_iter=n_power_iter,
            orthogonal_init=orthogonal_init,
        )
    )
    return nn.Sequential(*layers)


class KoopmanAESplit(nn.Module):
    """
    Koopman autoencoder with latent components [z_dyn, z_aux].
    The encoder and decoder use the full latent (dim_z_total); the linear
    Koopman operator acts only on z_dyn. Optional delay embedding changes
    the encoder input and time alignment; delay_L=0 uses individual frames.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        hidden_dims_enc=None,
        hidden_dims_dec=None,
        alpha: float = 0.99,
        sn_coeff: float = 1.0,
        sn_power_iter: int = 1,
        orthogonal_init: bool = True,
        dim_z_dyn: int = None,
        delay_L: int = 0,
    ):
        super().__init__()
        hidden_dims_enc = hidden_dims_enc if hidden_dims_enc is not None else [256, 256]
        hidden_dims_dec = hidden_dims_dec if hidden_dims_dec is not None else [256, 256]

        self.dim_x = dim_x
        self.dim_z = dim_z  # Total latent dimension.
        self.dim_z_dyn = dim_z_dyn if dim_z_dyn is not None else dim_z
        assert 0 < self.dim_z_dyn <= self.dim_z, "dim_z_dyn must be in (0, dim_z]"
        self.dim_z_aux = self.dim_z - self.dim_z_dyn
        self.delay_L = int(delay_L)
        assert self.delay_L >= 0, "delay_L must be non-negative"
        # Encoder input: dim_x for delay_L=0, otherwise (L+1)*dim_x.
        self.enc_in_dim = (self.delay_L + 1) * dim_x if self.delay_L > 0 else dim_x

        self.encoder = make_mlp(
            self.enc_in_dim,
            self.dim_z,
            hidden_dims_enc,
            coeff=sn_coeff,
            n_power_iter=sn_power_iter,
            orthogonal_init=orthogonal_init,
        )
        self.decoder = make_mlp(
            self.dim_z,
            dim_x,
            hidden_dims_dec,
            coeff=sn_coeff,
            n_power_iter=sn_power_iter,
            orthogonal_init=orthogonal_init,
        )

        self.alpha = alpha
        self.K_raw = nn.Parameter(torch.randn(self.dim_z_dyn, self.dim_z_dyn) * 0.01)
        self.register_buffer("K_u", torch.randn(1, self.dim_z_dyn))
        self.sn_power_iter = sn_power_iter
        # Shared switch for K_u and all SNLinear.u buffer updates.
        self.buffer_update_enabled = True

    def _spectral_norm_K(self):
        W = self.K_raw
        u = self.K_u
        n_iter = max(1, self.sn_power_iter)
        v = F.normalize(u @ W, dim=1, eps=1e-12)
        for _ in range(n_iter):
            v = F.normalize(u @ W, dim=1, eps=1e-12)
            u = F.normalize(v @ W.t(), dim=1, eps=1e-12)
        if self.buffer_update_enabled:
            self.K_u = u.detach()
        # Static mode uses power-iteration results without updating buffers.
        sigma = torch.sum(u @ W @ v.t())
        K = self.alpha * W / (sigma + 1e-12)
        return K

    def set_buffer_update(self, enabled: bool):
        # Single entry point used by diffusion training to freeze/unfreeze
        # spectral-normalization power-iteration buffers.
        self.buffer_update_enabled = bool(enabled)
        for m in self.modules():
            if isinstance(m, SNLinear):
                m.buffer_update_enabled = bool(enabled)

    def split_z(self, z):
        """
        Split z into (z_dyn, z_aux).
        z: [..., dim_z]
        Returns:
            z_dyn: [..., dim_z_dyn]
            z_aux: [..., dim_z_aux]; an empty [..., 0] tensor if dim_z_aux=0.
        """
        if self.dim_z_aux > 0:
            z_dyn = z[..., : self.dim_z_dyn]
            z_aux = z[..., self.dim_z_dyn :]
        else:
            z_dyn = z
            z_aux = z.new_zeros(z.shape[:-1] + (0,))
        return z_dyn, z_aux

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_seq, return_k: bool = False):
        """
        x_seq: [B, T, dim_x], the input time series.
        return_k: Return the K used in this forward pass for reuse within
            the same step, without updating K_u again.
        Returns:
            z_seq_full:  [B, T_eff, dim_z], the full encoded latent sequence.
            z_next_pred: [B, T_eff-1, dim_z_dyn], one-step predictions K z_t.
            z_next_true: [B, T_eff-1, dim_z_dyn], targets z_dyn at t+1.
            K(optional): [dim_z_dyn, dim_z_dyn], the operator used above.
        Shapes:
            T_eff = T              (delay_L == 0)
            T_eff = T - delay_L    (delay_L > 0)
        Note:
            With delay_L > 0, encoded z at index t corresponds to input
            x at t+delay_L; the encoder receives a delay-embedded window.
        """
        B, T, D = x_seq.shape
        assert D == self.dim_x, "Input dimension does not match dim_x"
        L = self.delay_L

        if L == 0:
            z_seq_full = self.encode(x_seq.reshape(B * T, D)).reshape(B, T, -1)
        else:
            assert T > L, "Time length T must exceed delay_L"
            T_eff = T - L
            chunks = []
            for i in range(L + 1):
                chunks.append(x_seq[:, i : i + T_eff, :])  # [B, T_eff, D]
            x_delay = torch.cat(chunks, dim=-1)  # [B, T_eff, (L+1)D]
            x_enc = x_delay.reshape(B * T_eff, self.enc_in_dim)
            z_seq_full = self.encode(x_enc).reshape(B, T_eff, -1)

        z_dyn_seq, _ = self.split_z(z_seq_full)
        K = self._spectral_norm_K()
        z_next_pred = torch.einsum("ij,btj->bti", K, z_dyn_seq[:, :-1])
        z_next_true = z_dyn_seq[:, 1:]
        if return_k:
            return z_seq_full, z_next_pred, z_next_true, K
        return z_seq_full, z_next_pred, z_next_true
