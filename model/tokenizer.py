"""
Kronos Tokenizer - Hybrid quantization for financial time series.
"""
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .module import TransformerBlock, BSQuantizer


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    Kronos Tokenizer for tokenizing input data using hybrid quantization.

    Uses encoder-decoder Transformer blocks with Binary Spherical Quantization (BSQuantizer).
    """

    def __init__(
        self,
        d_in: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        ff_dim: int = 1024,
        n_enc_layers: int = 3,
        n_dec_layers: int = 3,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
        s1_bits: int = 8,
        s2_bits: int = 8,
        beta: float = 0.25,
        gamma0: float = 1.0,
        gamma: float = 1.0,
        zeta: float = 1.0,
        group_size: int = 8,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits

        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])

        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size)

    def forward(self, x: torch.Tensor):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        if half:
            x1, x2 = x[0], x[1]
            mask = 2 ** torch.arange(self.codebook_dim // 2, device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0

        x = x.float() * 2 - 1
        q_scale = 1.0 / (self.codebook_dim ** 0.5)
        x = x * q_scale
        return x

    def encode(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z