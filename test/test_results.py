"""Unit tests for qaoa_parameter_setting/utils/results.py"""

import pytest
import numpy as np
from qaoa_parameter_setting.utils.results import (
    sanitize_energy,
    result_contains_noopt,
)

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestSanitizeEnergy:
    """Tests for sanitize_energy function."""

    def test_float_energy(self):
        """Test sanitizing float energy values."""
        assert sanitize_energy(1.5) == 1.5
        assert sanitize_energy(0.0) == 0.0
        assert sanitize_energy(-10.5) == -10.5

    def test_none_energy(self):
        """Test sanitizing None energy values."""
        assert sanitize_energy(None) is None

    def test_na_string_energy(self):
        """Test sanitizing 'NA' string energy values."""
        assert sanitize_energy("NA") is None

    def test_numpy_float_energy(self):
        """Test sanitizing numpy float energy values."""
        np_float = np.float64(3.14)
        result = sanitize_energy(np_float)
        assert result == pytest.approx(3.14)
        
        np_float32 = np.float32(2.71)
        result = sanitize_energy(np_float32)
        assert result == pytest.approx(2.71)

    def test_unknown_string_raises_error(self):
        """Test that unknown string values raise ValueError."""
        with pytest.raises(ValueError, match="Unknown energy value"):
            sanitize_energy("unknown")
        
        with pytest.raises(ValueError, match="Unknown energy value"):
            sanitize_energy("NaN")
        
        with pytest.raises(ValueError, match="Unknown energy value"):
            sanitize_energy("inf")

    def test_unknown_type_raises_error(self):
        """Test that unknown types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown energy value type"):
            sanitize_energy([1.5])
        
        with pytest.raises(ValueError, match="Unknown energy value type"):
            sanitize_energy({"energy": 1.5})
        
        with pytest.raises(ValueError, match="Unknown energy value type"):
            sanitize_energy(123)  # int is not accepted

    def test_special_float_values(self):
        """Test sanitizing special float values."""
        assert sanitize_energy(float('inf')) == float('inf')
        assert sanitize_energy(float('-inf')) == float('-inf')
        # NaN is tricky - it's not equal to itself
        result = sanitize_energy(float('nan'))
        assert np.isnan(result)

    def test_numpy_special_values(self):
        """Test sanitizing numpy special values."""
        assert sanitize_energy(np.inf) == np.inf
        assert sanitize_energy(-np.inf) == -np.inf
        result = sanitize_energy(np.nan)
        assert np.isnan(result)


class TestResultContainsNoopt:
    """Tests for result_contains_noopt function."""

    def test_tqa_opt_returns_true(self):
        """Test that TQA_opt configs return True."""
        result = {
            "args": {
                "config": "configs/TQA_SV_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_fa_opt_returns_true(self):
        """Test that FA_opt configs return True."""
        result = {
            "args": {
                "config": "configs/FA_MPS_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_faaer_opt_returns_true(self):
        """Test that FAAer_opt configs return True."""
        result = {
            "args": {
                "config": "configs/FAAer_SV_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_tqaaer_opt_returns_true(self):
        """Test that TQAAer_opt configs return True."""
        result = {
            "args": {
                "config": "configs/TQAAer_MPS_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_no_opt_config_returns_false(self):
        """Test that no_opt configs return False."""
        result = {
            "args": {
                "config": "configs/FA_SV_no_opt.json"
            }
        }
        assert result_contains_noopt(result) is False

    def test_noopt_config_returns_false(self):
        """Test that noOpt configs return False."""
        result = {
            "args": {
                "config": "configs/TQA_SV_no_pt.json"
            }
        }
        assert result_contains_noopt(result) is False

    def test_other_method_returns_false(self):
        """Test that other methods return False."""
        result = {
            "args": {
                "config": "configs/F_SV_opt.json"
            }
        }
        assert result_contains_noopt(result) is False
        
        result = {
            "args": {
                "config": "configs/LR_MPS_opt.json"
            }
        }
        assert result_contains_noopt(result) is False

    def test_method_without_opt_returns_false(self):
        """Test that methods without 'opt' return False."""
        result = {
            "args": {
                "config": "configs/TQA_SV.json"
            }
        }
        assert result_contains_noopt(result) is False

    def test_case_sensitivity(self):
        """Test case sensitivity in method detection."""
        # The function converts to lowercase, so OPT matches opt
        result = {
            "args": {
                "config": "configs/FA_SV_OPT.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_with_path_separators(self):
        """Test with different path separators."""
        # Windows path separator causes issues with Path.parts
        result = {
            "args": {
                "config": "configs\\FA_SV_opt.json"
            }
        }
        # This will fail because backslash is not handled correctly
        assert result_contains_noopt(result) is False
        
        result = {
            "args": {
                "config": "/path/to/configs/TQA_MPS_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_config_filename_only(self):
        """Test with just the config filename."""
        result = {
            "args": {
                "config": "FA_SV_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_multiple_underscores_in_config(self):
        """Test configs with multiple underscores."""
        result = {
            "args": {
                "config": "configs/FA_SV_custom_opt.json"
            }
        }
        assert result_contains_noopt(result) is True


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_sanitize_energy_with_very_small_float(self):
        """Test sanitizing very small float values."""
        small_value = 1e-100
        assert sanitize_energy(small_value) == small_value

    def test_sanitize_energy_with_very_large_float(self):
        """Test sanitizing very large float values."""
        large_value = 1e100
        assert sanitize_energy(large_value) == large_value

    def test_sanitize_energy_with_negative_zero(self):
        """Test sanitizing negative zero."""
        assert sanitize_energy(-0.0) == 0.0

    def test_result_contains_noopt_with_empty_config(self):
        """Test result_contains_noopt with empty config string."""
        result = {
            "args": {
                "config": ""
            }
        }
        # Empty config causes IndexError
        with pytest.raises(IndexError):
            result_contains_noopt(result)

    def test_result_contains_noopt_with_missing_extension(self):
        """Test result_contains_noopt with config missing .json extension."""
        result = {
            "args": {
                "config": "FA_SV_opt"
            }
        }
        assert result_contains_noopt(result) is True

    def test_result_contains_noopt_with_multiple_dots(self):
        """Test result_contains_noopt with multiple dots in filename."""
        result = {
            "args": {
                "config": "configs/FA.SV.opt.json"
            }
        }
        # Splits on first dot, so "FA" is extracted, which is not in the list
        assert result_contains_noopt(result) is False

    def test_sanitize_energy_na_case_sensitivity(self):
        """Test that 'NA' is case-sensitive."""
        with pytest.raises(ValueError, match="Unknown energy value"):
            sanitize_energy("na")
        
        with pytest.raises(ValueError, match="Unknown energy value"):
            sanitize_energy("Na")
        
        with pytest.raises(ValueError, match="Unknown energy value"):
            sanitize_energy("nA")

    def test_result_contains_noopt_with_angle_opt(self):
        """Test result_contains_noopt with angle_opt configs."""
        result = {
            "args": {
                "config": "configs/LR_SV_angle_opt.json"
            }
        }
        # LR is not in the no_opt_matches list
        assert result_contains_noopt(result) is False

    def test_result_contains_noopt_all_caps_method(self):
        """Test result_contains_noopt with all caps method names."""
        result = {
            "args": {
                "config": "configs/TQA_SV_OPT.json"
            }
        }
        # The function converts to lowercase, so OPT matches opt
        assert result_contains_noopt(result) is True

    def test_sanitize_energy_with_numpy_array(self):
        """Test that numpy arrays raise an error."""
        with pytest.raises(ValueError, match="Unknown energy value type"):
            sanitize_energy(np.array([1.5]))

    def test_result_contains_noopt_with_nested_path(self):
        """Test result_contains_noopt with deeply nested path."""
        result = {
            "args": {
                "config": "a/b/c/d/e/FA_SV_opt.json"
            }
        }
        assert result_contains_noopt(result) is True

    def test_result_contains_noopt_fa_and_tqa_together(self):
        """Test result_contains_noopt when both FA and TQA appear in name."""
        result = {
            "args": {
                "config": "configs/FA_TQA_SV_opt.json"
            }
        }
        # Should still return True as FA is in the list
        assert result_contains_noopt(result) is True