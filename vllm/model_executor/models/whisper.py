"""Minimal implementation of CLIPVisionModel intended to be only used 
within a Qwen-Audio model."""
from typing import Dict, Iterable, Optional, List

import torch
from torch import Tensor, nn
from transformers import WhisperConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 bias: bool = True,
                 config: Optional[WhisperConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 cache_config: Optional[CacheConfig] = None,):
        super().__init__()
        self.n_head = num_heads
        self.k_projj = RowParallelLinear(input_size=hidden_size,
                                         output_size=hidden_size,
                                         bias=False,
                                         quant_config=quant_config
                                         )
        self.v_proj = RowParallelLinear(input_size=hidden_size,
                                        output_size=hidden_size,
                                        bias=bias,
                                        quant_config=quant_config
                                        )
        self.q_proj = RowParallelLinear(input_size=hidden_size,
                                        output_size=hidden_size,
                                        bias=bias,
                                        quant_config=quant_config
                                        )
        self.out_proj = RowParallelLinear(input_size=hidden_size,
                                          output_size=hidden_size,
                                          bias=bias,
                                          quant_config=quant_config
                                          )


class AudioEncoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_ctx: int,
            n_state: int,
            n_head: int,
            n_layer: int,
            output_dim: int = 512,
            avg_pool: bool = True,
            add_audio_bos_eos_token: bool = True,
            **kwargs
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

        if avg_pool:
            self.avg_pooler = nn.AvgPool1d(2, stride=2)
        else:
            self.avg_pooler = None
        self.proj = nn.Linear(n_state, output_dim)
        if add_audio_bos_eos_token:
            self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        else:
            self.audio_bos_eos_token = None
        self.output_dim = output_dim
        self.n_head = n_head