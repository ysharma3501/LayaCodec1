""" Vocos (see https://arxiv.org/abs/2306.00814)."""

# Added upsampler blocks and custom istft head
# Adapted from:
# https://github.com/lucadellalib/focalcodec/blob/main/focalcodec/vocos.py

import warnings
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch import Tensor, nn
from typing import Any, List, Optional, Tuple
from torch.nn.utils.parametrizations import weight_norm

__all__ = ["Vocos"]

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
      
class UpSamplerBlock(nn.Module):
    """Transpose Conv plus Resnet Blocks to upsample feature embedding."""
    def __init__(self, in_channels: int, upsample_factors: List[int], kernel_sizes: Optional[List[int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factors = list(upsample_factors or [])
        self.kernel_sizes = list(kernel_sizes or [8] * len(self.upsample_factors))

        assert len(self.kernel_sizes) == len(self.upsample_factors), "kernel_sizes and upsample_factors must have the same length"

        self.upsample_layers = nn.ModuleList()
        self.resnet_blocks  = nn.ModuleList()
        self.out_proj = nn.Linear(self.in_channels // (2 ** len(self.upsample_factors)), self.in_channels, bias=True)

        for i, (k, u) in enumerate(zip(self.kernel_sizes, self.upsample_factors)):
            c_in  = self.in_channels // (2 ** i)
            c_out = self.in_channels // (2 ** (i + 1))
            self.upsample_layers.append(
                weight_norm(nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=u, padding=(k - u) // 2))
            )
            self.resnet_blocks.append(
                ResnetBlock(in_channels=c_out, out_channels=c_out, dropout=0.0, temb_channels=0)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> ... -> [B, C', L']
        for up, rsblk in zip(self.upsample_layers, self.resnet_blocks):
            x = rsblk(up(x))
        # [B, C', L'] -> [B, L', C]（元の hidden_dim に戻す）
        return nonlinearity(self.out_proj(x.transpose(1, 2)))
      

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for 1D input.

    Parameters
    ----------
    dim:
        Number of input channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    layerscale_init:
        Initial value for layer scaling parameter.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        layerscale_init: "Optional[float]" = None,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.layerscale_init = layerscale_init
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=0 if causal else "same", groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(ffn_dim, dim)

        # Parameters
        if layerscale_init is not None:
            self.gamma = nn.Parameter(
                torch.full((dim,), layerscale_init),
            )
        else:
            self.gamma = None

    def forward(
        self,
        input: "Tensor",
        left_context: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        left_context:
            Left context of shape (batch_size, kernel_size - 1, dim).
            If None, initialized as zeros.

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated left context for next chunk.

        """
        input = input.permute(0, 2, 1)
        orig_input = input
        if self.causal:
            if left_context is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context = left_context.permute(0, 2, 1)
                input = torch.cat([left_context, input], dim=-1)
            left_context = input[..., -self.causal_pad :].permute(0, 2, 1)
        else:
            left_context = None
        output = self.dwconv(input)
        output = output.permute(0, 2, 1)
        output = self.norm(output)
        output = self.pwconv1(output)
        output = self.activation(output)
        output = self.pwconv2(output)
        if self.gamma is not None:
            output = self.gamma * output
        output = output.permute(0, 2, 1)
        output = orig_input + output
        output = output.permute(0, 2, 1)

        return output, left_context


class VocosBackbone(nn.Module):
    """Vocos backbone.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of hidden channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    layerscale_init:
        Initial value for layer scaling parameter.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        layerscale_init: "Optional[float]" = None,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.layerscale_init = (
            1 / num_layers if layerscale_init is None else layerscale_init
        )
        self.causal = causal
        self.causal_pad = kernel_size - 1

        # Modules
        self.embedding = nn.Conv1d(
            input_dim, dim, kernel_size, padding=0 if causal else "same"
        )
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.layers = nn.ModuleList(
            ConvNeXtBlock(dim, ffn_dim, kernel_size, self.layerscale_init, causal)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        left_contexts:
            Left contexts for each layer.
            If provided, the first tensor should be of shape (batch_size, kernel_size - 1, input_dim),
            the following tensors should be of shape (batch_size, kernel_size - 1, dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length, dim);
            - updated left contexts for each layer.

        """
        input = input.permute(0, 2, 1)
        left_context_embedding = None if left_contexts is None else left_contexts[0]
        if self.causal:
            if left_context_embedding is None:
                input = nn.functional.pad(input, [self.causal_pad, 0], mode="replicate")
            else:
                left_context_embedding = left_context_embedding.permute(0, 2, 1)
                input = torch.cat([left_context_embedding, input], dim=-1)
            new_left_context = input[..., -self.causal_pad :].permute(0, 2, 1)
            new_left_contexts: List[Optional[Tensor]] = [new_left_context]
        else:
            new_left_contexts: List[Optional[Tensor]] = [None]
        output = self.embedding(input)
        output = output.permute(0, 2, 1)
        output = self.input_norm(output)
        for i, layer in enumerate(self.layers):
            output, new_left_context = layer(
                output,
                None if left_contexts is None else left_contexts[i + 1],
            )
            new_left_contexts.append(new_left_context)
        output = self.output_norm(output)
        return output, new_left_contexts


class ISTFT(nn.Module):
    """Custom implementation of inverse STFT with support for non-causal and causal padding.

    Parameters
    ----------
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    win_length:
        Size of window frame and STFT filter.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        win_length: "int" = 1024,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.causal = causal
        self.non_causal_pad = (win_length - hop_length) // 2

        # Buffers (JIT compilable)
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("window_sq", window**2, persistent=False)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, n_fft // 2 + 1).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        # Inverse FFT
        input = input.permute(0, 2, 1)
        ifft = torch.fft.irfft(input, self.n_fft, dim=1, norm="backward")

        if self.causal:
            output = ifft[:, -self.hop_length :].permute(0, 2, 1)
            output = output.flatten(start_dim=1)
            return output

        # Overlap-add
        T = input.shape[-1]
        ifft = ifft * self.window[None, :, None]
        output_size = int((T - 1) * self.hop_length + self.win_length)
        output = nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, self.non_causal_pad : -self.non_causal_pad]

        # Window envelope
        window_sq = self.window_sq.expand(1, T, -1).permute(0, 2, 1)
        window_envelope = nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[0, 0, 0, self.non_causal_pad : -self.non_causal_pad]

        # Normalize
        output /= window_envelope

        return output

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}("
            f"n_fft={self.n_fft}, "
            f"hop_length={self.hop_length}, "
            f"win_length={self.win_length})"
        )


class ISTFTHead(nn.Module):
    """Inverse STFT head.

    Parameters
    ----------
    dim:
        Number of input channels.
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        dim: "int" = 512,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.causal = causal

        # Modules
        self.proj = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, n_fft, causal)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        output = self.proj(input)
        mag, phase = output.chunk(2, dim=-1)
        mag = mag.exp()
        # Safeguard to prevent excessively large magnitudes
        mag = mag.clamp(max=1e2)
        # Real and imaginary value
        # JIT compilable
        stft = mag * torch.complex(phase.cos(), phase.sin())
        output = self.istft(stft)
        return output


class Vocos(nn.Module):
    """Vocos generator for waveform synthesis.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of input channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    layerscale_init:
        Initial value for layer scaling parameter.
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    causal:
        Whether the module should be causal.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        layerscale_init: "Optional[float]" = None,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        causal: "bool" = False,
        **kwargs: "Any",
    ) -> "None":
        super().__init__()
        if kwargs.get("input_channels", None) is not None:
            warnings.warn(
                "`input_channels` is deprecated, please use `input_dim` instead",
                DeprecationWarning,
            )
            input_dim = kwargs["input_channels"]
        if kwargs.get("padding", None) is not None:
            warnings.warn(
                "`padding` is deprecated and no longer used. It is now automatically set "
                'to "causal" if the model is causal, "same" otherwise',
                DeprecationWarning,
            )

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.layerscale_init = layerscale_init or 1 / num_layers
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.causal = causal
        self.upsample_factor = hop_length
        self.chunk_size = 1

        # Modules
        self.backbone = VocosBackbone(
            input_dim,
            num_layers,
            dim,
            ffn_dim,
            kernel_size,
            layerscale_init,
            causal,
        )
        self.head = ISTFTHead(dim, n_fft, hop_length, causal)
        self.upsampler = UpSamplerBlock(in_channels=512,
                                            upsample_factors=[3, 3],
                                            kernel_sizes=[9, 9])

    def forward(
        self,
        input: "Tensor",
        left_contexts: "Optional[List[Optional[Tensor]]]" = None,
    ) -> "Tuple[Tensor, List[Optional[Tensor]]]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        left_contexts:
            Left contexts for each backbone layer.
            If provided, the first tensor should be of shape (batch_size, kernel_size - 1, input_dim),
            the following tensors should be of shape (batch_size, kernel_size - 1, dim).

        Returns
        -------
            - Output tensor of shape (batch_size, seq_length * hop_length);
            - updated left contexts for each backbone layer.

        """
        output, left_contexts = self.backbone(input, left_contexts)
        output = self.upsampler(output.transpose(1,2))
        output = self.head(output)
        return output, left_contexts
