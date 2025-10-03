from main import multiple_of


class TestMultipleOf:
    """Tests for the multiple_of function"""

    def test_exact_multiple(self):
        """Test when n is already an exact multiple of factor"""
        assert multiple_of(256, 512) == 512
        assert multiple_of(10, 100) == 100
        assert multiple_of(5, 25) == 25

    def test_round_up(self):
        """Test rounding up to nearest multiple"""
        assert multiple_of(256, 300) == 256
        assert multiple_of(10, 15) == 20
        assert multiple_of(8, 11) == 8

    def test_round_down(self):
        """Test rounding down to nearest multiple"""
        assert multiple_of(256, 200) == 256
        assert multiple_of(10, 14) == 10
        assert multiple_of(8, 5) == 8

    def test_midpoint_rounds_to_even(self):
        """Test standard rounding behavior at midpoint"""
        # Python's round() uses banker's rounding (round half to even)
        assert multiple_of(10, 25) == 20  # 25/10 = 2.5, rounds to even (2)
        assert multiple_of(10, 35) == 40  # 35/10 = 3.5, rounds to even (4)

    def test_float_input(self):
        """Test with float values for n"""
        assert multiple_of(256, 300.5) == 256
        assert multiple_of(10, 17.8) == 20
        assert multiple_of(8, 9.2) == 8

    def test_small_numbers(self):
        """Test with small values"""
        assert multiple_of(2, 1) == 2
        assert multiple_of(2, 3) == 4
        assert multiple_of(5, 1) == 5

    def test_large_numbers(self):
        """Test with large values"""
        assert multiple_of(256, 10000) == 9984  # 10000/256 = 39.0625, rounds to 39
        assert multiple_of(1000, 12345) == 12000

    def test_swiglu(self):
        """Test realistic use case from SwiGLU initialization"""
        d_hidden = 512
        expansion_factor = 4.0
        n = d_hidden * expansion_factor * 2 / 3
        assert multiple_of(256, n) == 1280

    def test_factor_one(self):
        """Test when factor is 1"""
        assert multiple_of(1, 5) == 5
        assert multiple_of(1, 5.7) == 6
        assert multiple_of(1, 5.3) == 5
