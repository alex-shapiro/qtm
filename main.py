from dataclasses import dataclass
from typing import final, override

from torch import Tensor, nn
from torch._prims_common import DeviceLikeType
import torch.nn.functional as F


@dataclass
class ModelConfig:
    device: DeviceLikeType
    d_in: int
    d_out: int
    d_hidden: int = 256
    num_heads: int = 8
    num_prediction_blocks: int = 8
    swiglu_expansion_factor: float = 4.0
    rms_norm_epsilon: float = 1e-5


@final
class QTestModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.action_predictor = ActionPredictor(config)
        self.q_head = nn.Linear(config.d_hidden, 2, device=config.device)


@final
class ActionPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.blocks = [
            PredictionBlock(config) for _ in range(config.num_prediction_blocks)
        ]

    # def forward(self, x: Tensor) -> Tensor:


@final
class PredictionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_hidden, num_heads=config.num_heads
        )
        self.activation = SwiGLU(config)
        self.rms_norm = nn.RMSNorm(
            normalized_shape=[], eps=config.rms_norm_epsilon
        )  # todo: normalized_shape

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        x = self.attention(hidden)
        x = self.activation(x)
        x = self.rms_norm(x)
        return x


@final
class SwiGLU(nn.Module):
    """Swish (SiLU) + GLU activation"""

    def __init__(self, config: ModelConfig):
        super().__init__()  # _pyright: ignore[reportUnknownMemberType]
        d_internal = multiple_of(
            256, config.d_hidden * config.swiglu_expansion_factor * 2 / 3
        )
        self.up_gate_projections = nn.Linear(
            config.d_hidden, d_internal * 2, bias=False, device=config.device
        )
        self.down_projection = nn.Linear(
            d_internal, config.d_hidden, bias=False, device=config.device
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.up_gate_projections(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_projection(x)
        return x


def multiple_of(factor: int, n: int | float) -> int:
    """
    Returns the closet number to n that is a multiple of `factor`.
    Both `factor` and `n` must be non-negative.
    """
    assert n > 0 and factor > 0
    return max(factor, round(n / factor) * factor)


def main():
    print("Hello from qtm!")


if __name__ == "__main__":
    main()
