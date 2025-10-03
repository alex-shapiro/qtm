import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ModelConfig, SwiGLU


class TestSwiGLU:
    """Tests for the SwiGLU activation module"""

    @pytest.fixture
    def config(self):
        """Default config for testing"""
        return ModelConfig(
            device="cpu",
            d_in=128,
            d_out=64,
            d_hidden=256,
            num_heads=8,
            num_prediction_blocks=4,
            swiglu_expansion_factor=4.0,
            rms_norm_epsilon=1e-5,
        )

    def test_initialization(self, config):
        """Test SwiGLU module initializes correctly"""
        swiglu = SwiGLU(config)
        assert swiglu is not None
        assert hasattr(swiglu, "up_gate_projections")
        assert hasattr(swiglu, "down_projection")

    def test_internal_dimension_calculation(self, config):
        """Test that d_internal is calculated correctly as multiple of 256"""
        swiglu = SwiGLU(config)
        # d_internal = multiple_of(256, 256 * 4.0 * 2/3) = multiple_of(256, 682.67) = 768
        expected_d_internal = 768
        # up_gate_projections projects to d_internal * 2
        assert swiglu.up_gate_projections.out_features == expected_d_internal * 2

    def test_projection_shapes(self, config):
        """Test that projection layers have correct input/output dimensions"""
        swiglu = SwiGLU(config)
        assert swiglu.up_gate_projections.in_features == config.d_hidden
        assert swiglu.down_projection.out_features == config.d_hidden
        # down_projection input should match d_internal
        assert swiglu.down_projection.in_features == swiglu.up_gate_projections.out_features // 2

    def test_forward_output_shape(self, config):
        """Test that forward pass produces correct output shape"""
        swiglu = SwiGLU(config)
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.d_hidden)
        output = swiglu(x)
        assert output.shape == (batch_size, seq_len, config.d_hidden)

    def test_forward_single_sample(self, config):
        """Test forward pass with a single sample"""
        swiglu = SwiGLU(config)
        x = torch.randn(1, config.d_hidden)
        output = swiglu(x)
        assert output.shape == (1, config.d_hidden)

    def test_forward_preserves_batch_dimensions(self, config):
        """Test that forward pass preserves batch dimensions"""
        swiglu = SwiGLU(config)
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, config.d_hidden)
            output = swiglu(x)
            assert output.shape[0] == batch_size

    def test_no_bias_in_up_gate(self, config):
        """Test that up_gate_projections has no bias"""
        swiglu = SwiGLU(config)
        assert swiglu.up_gate_projections.bias is None

    def test_gate_and_up_split(self, config):
        """Test that up_gate_projections output splits correctly"""
        swiglu = SwiGLU(config)
        x = torch.randn(2, config.d_hidden)
        up_gate_output = swiglu.up_gate_projections(x)
        gate, up = up_gate_output.chunk(2, dim=-1)
        # Both chunks should have same size
        assert gate.shape == up.shape
        # Each chunk should be half of up_gate_projections output
        assert gate.shape[-1] == up_gate_output.shape[-1] // 2

    def test_output_is_not_nan(self, config):
        """Test that forward pass doesn't produce NaN values"""
        swiglu = SwiGLU(config)
        x = torch.randn(4, config.d_hidden)
        output = swiglu(x)
        assert not torch.isnan(output).any()

    def test_output_is_not_inf(self, config):
        """Test that forward pass doesn't produce infinite values"""
        swiglu = SwiGLU(config)
        x = torch.randn(4, config.d_hidden)
        output = swiglu(x)
        assert not torch.isinf(output).any()

    def test_different_expansion_factors(self):
        """Test with different expansion factors"""
        for expansion_factor in [2.0, 4.0, 8.0]:
            config = ModelConfig(
                device="cpu",
                d_in=128,
                d_out=64,
                d_hidden=256,
                swiglu_expansion_factor=expansion_factor,
            )
            swiglu = SwiGLU(config)
            x = torch.randn(2, config.d_hidden)
            output = swiglu(x)
            assert output.shape == (2, config.d_hidden)

    def test_different_hidden_dimensions(self):
        """Test with different hidden dimensions"""
        for d_hidden in [128, 256, 512, 1024]:
            config = ModelConfig(
                device="cpu",
                d_in=128,
                d_out=64,
                d_hidden=d_hidden,
            )
            swiglu = SwiGLU(config)
            x = torch.randn(2, d_hidden)
            output = swiglu(x)
            assert output.shape == (2, d_hidden)

    def test_gradient_flow(self, config):
        """Test that gradients flow through the module"""
        swiglu = SwiGLU(config)
        x = torch.randn(2, config.d_hidden, requires_grad=True)
        output = swiglu(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
