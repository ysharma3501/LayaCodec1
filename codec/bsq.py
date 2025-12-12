"""Binary spherical quantization (see https://arxiv.org/abs/2406.07548)."""

# Adapted from:
# https://github.com/lucadellalib/focalcodec/blob/main/focalcodec/bsq.py

import math
from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = ["BinarySphericalQuantizer"]


class BinarySphericalQuantizer(nn.Module):
    """Binary spherical quantizer that maps inputs to binary codes on the unit hypersphere.

    Parameters
    ----------
    codebook_size:
        Number of binary codes in the codebook.

    """

    def __init__(self, codebook_size: "int" = 4096) -> "None":
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = int(math.log2(codebook_size))

        # Buffers
        self.register_buffer(
            "codebook_value",
            torch.tensor(1 / math.sqrt(self.dim)),
            persistent=False,
        )
        self.register_buffer(
            "mask", 2 ** torch.arange(self.dim - 1, -1, -1), persistent=False
        )
        all_codes = torch.arange(codebook_size)
        bits = (all_codes[..., None].int() & self.mask) != 0
        codebook = self._bits_to_codes(bits) * self.codebook_value
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, lats: "Tensor") -> "Tuple[Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            - Output tokens of shape (...);
            - output codes (i.e. quantized latents) of shape (..., dim).

        """
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        return toks, codes

    @torch.jit.export
    def lats_to_codes(self, lats: "Tensor") -> "Tensor":
        """Transform latents into codes (i.e. quantized latents).

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            Output codes of shape (..., dim).

        """
        return torch.where(lats > 0, self.codebook_value, -self.codebook_value)

    @torch.jit.export
    def lats_to_toks(self, lats: "Tensor") -> "Tensor":
        """Transform latents into tokens.

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            Output tokens of shape (...).

        """
        return self.codes_to_toks(lats)

    @torch.jit.export
    def codes_to_toks(self, codes: "Tensor") -> "Tensor":
        """Transform codes (i.e. quantized latents) into tokens.

        Parameters
        ----------
        codes:
            Input codes of shape (..., dim).

        Returns
        -------
            Output tokens of shape (...).

        """
        return ((codes > 0) * self.mask).sum(dim=-1)

    @torch.jit.export
    def toks_to_codes(self, toks: "Tensor") -> "Tensor":
        """Transform tokens into codes (i.e. quantized latents).

        Parameters
        ----------
        toks:
            Input tokens of shape (...).

        Returns
        -------
            Output codes of shape (..., dim).

        """
        # ONNX compilable
        bits = ((toks[..., None] // self.mask) % 2).to(self.codebook.dtype)
        return self._bits_to_codes(bits) * self.codebook_value

    def _bits_to_codes(self, bits: "Tensor") -> "Tensor":
        return bits * 2 - 1

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(codebook_size={self.codebook_size})"
