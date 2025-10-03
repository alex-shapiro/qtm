import pytest
import torch

from rope import RotaryPositionalEmbeddings


class TestRotaryPositionalEmbeddings:
    """Minimal sanity checks for RoPE implementation"""

    def test_initialization(self):
        """Test RoPE module initializes correctly"""
        rope = RotaryPositionalEmbeddings(dim=64, max_seq_len=2048, base=10_000)
        assert rope.dim == 64
        assert rope.max_seq_len == 2048
        assert rope.base == 10_000
        assert hasattr(rope, "theta")
        assert hasattr(rope, "cache")

    def test_theta_shape(self):
        """Test theta buffer has correct shape"""
        dim = 64
        rope = RotaryPositionalEmbeddings(dim=dim)
        assert rope.theta.shape == (dim // 2,)

    def test_cache_shape(self):
        """Test cache buffer has correct shape"""
        dim = 64
        max_seq_len = 2048
        rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len)
        assert rope.cache.shape == (max_seq_len, dim // 2, 2)

    def test_forward_output_shape(self):
        """Test forward pass preserves input shape"""
        batch_size = 4
        seq_len = 128
        num_heads = 8
        head_dim = 64

        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=2048)
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        output = rope(x)

        assert output.shape == x.shape

    def test_forward_with_input_pos(self):
        """Test forward pass with explicit input positions"""
        batch_size = 2
        seq_len = 10
        num_heads = 4
        head_dim = 32

        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=100)
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        input_pos = torch.arange(seq_len)
        output = rope(x, input_pos=input_pos)

        assert output.shape == x.shape

    def test_output_dtype_matches_input(self):
        """Test output has same dtype as input"""
        rope = RotaryPositionalEmbeddings(dim=64)

        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(2, 10, 4, 64, dtype=dtype)
            output = rope(x)
            assert output.dtype == dtype

    def test_output_is_not_nan(self):
        """Test forward pass doesn't produce NaN values"""
        rope = RotaryPositionalEmbeddings(dim=64)
        x = torch.randn(2, 10, 4, 64)
        output = rope(x)
        assert not torch.isnan(output).any()

    def test_output_is_not_inf(self):
        """Test forward pass doesn't produce infinite values"""
        rope = RotaryPositionalEmbeddings(dim=64)
        x = torch.randn(2, 10, 4, 64)
        output = rope(x)
        assert not torch.isinf(output).any()

    def test_different_dimensions(self):
        """Test with different head dimensions"""
        for dim in [32, 64, 128]:
            rope = RotaryPositionalEmbeddings(dim=dim)
            x = torch.randn(2, 10, 4, dim)
            output = rope(x)
            assert output.shape == x.shape

    def test_cache_rebuild(self):
        """Test that cache can be rebuilt for longer sequences"""
        rope = RotaryPositionalEmbeddings(dim=64, max_seq_len=100)
        initial_cache_shape = rope.cache.shape

        rope.build_rope_cache(max_seq_len=200)
        new_cache_shape = rope.cache.shape

        assert new_cache_shape[0] == 200
        assert new_cache_shape[1:] == initial_cache_shape[1:]

    def test_reset_parameters(self):
        """Test reset_parameters reinitializes the module"""
        rope = RotaryPositionalEmbeddings(dim=64, max_seq_len=100)
        original_theta = rope.theta.clone()

        rope.reset_parameters()

        # Theta should be recomputed to same values
        assert torch.allclose(rope.theta, original_theta)

    def test_inference_mode_single_position(self):
        """Test inference mode with single position"""
        batch_size = 1
        seq_len = 1
        num_heads = 4
        head_dim = 64

        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=100)
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        input_pos = torch.tensor([50])  # Position 50

        output = rope(x, input_pos=input_pos)
        assert output.shape == x.shape
