"""Unit tests for qaoa_parameter_setting/utils/metadata.py"""

import pytest
from unittest.mock import Mock, patch
from qaoa_parameter_setting.utils.metadata import (
    total_runtime,
    guess_problem_class,
    guess_problem_class_from_filename,
)

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestTotalRuntime:
    """Tests for total_runtime function."""

    def test_simple_runtime_calculation(self):
        """Test calculating total runtime from simple data."""
        data = {
            "iteration_0": {"train_duration": 10.5},
            "iteration_1": {"train_duration": 15.2},
            "iteration_2": {"train_duration": 8.3},
        }
        assert total_runtime(data) == pytest.approx(34.0)

    def test_excludes_args(self):
        """Test that 'args' key is excluded from runtime calculation."""
        data = {
            "args": {"train_duration": 100.0},
            "iteration_0": {"train_duration": 10.0},
        }
        assert total_runtime(data) == pytest.approx(10.0)

    def test_excludes_pre_processing(self):
        """Test that 'pre_processing' key is excluded from runtime calculation."""
        data = {
            "pre_processing": {"train_duration": 50.0},
            "iteration_0": {"train_duration": 10.0},
        }
        assert total_runtime(data) == pytest.approx(10.0)

    def test_excludes_cost_operator(self):
        """Test that 'cost_operator' key is excluded from runtime calculation."""
        data = {
            "cost_operator": {"train_duration": 75.0},
            "iteration_0": {"train_duration": 10.0},
        }
        assert total_runtime(data) == pytest.approx(10.0)

    def test_excludes_all_special_keys(self):
        """Test that all special keys are excluded."""
        data = {
            "args": {"train_duration": 100.0},
            "pre_processing": {"train_duration": 50.0},
            "cost_operator": {"train_duration": 75.0},
            "iteration_0": {"train_duration": 10.0},
            "iteration_1": {"train_duration": 20.0},
        }
        assert total_runtime(data) == pytest.approx(30.0)

    def test_empty_data(self):
        """Test with empty data dictionary."""
        data = {}
        assert total_runtime(data) == 0

    def test_only_excluded_keys(self):
        """Test with only excluded keys."""
        data = {
            "args": {"train_duration": 100.0},
            "pre_processing": {"train_duration": 50.0},
        }
        assert total_runtime(data) == 0

    def test_zero_durations(self):
        """Test with zero durations."""
        data = {
            "iteration_0": {"train_duration": 0.0},
            "iteration_1": {"train_duration": 0.0},
        }
        assert total_runtime(data) == 0.0

    def test_large_number_of_iterations(self):
        """Test with many iterations."""
        data = {f"iteration_{i}": {"train_duration": 1.0} for i in range(100)}
        assert total_runtime(data) == pytest.approx(100.0)


class TestGuessProblemClassFromFilename:
    """Tests for guess_problem_class_from_filename function."""

    def test_mc_problem_class(self):
        """Test identifying MC (Max Cut) problem class from filename."""
        assert guess_problem_class_from_filename("result_MC_graph.json") == "MC"
        assert guess_problem_class_from_filename("path/to/data_MC_instance.json") == "MC"
        assert guess_problem_class_from_filename("prefix_MC_suffix.json") == "MC"

    def test_mis_problem_class(self):
        """Test identifying MIS (Maximum Independent Set) problem class from filename."""
        assert guess_problem_class_from_filename("result_MIS_graph.json") == "MIS"
        assert guess_problem_class_from_filename("path/to/data_MIS_instance.json") == "MIS"
        assert guess_problem_class_from_filename("prefix_MIS_suffix.json") == "MIS"

    def test_no_problem_class_returns_none(self):
        """Test that None is returned when no problem class is found."""
        assert guess_problem_class_from_filename("result_graph.json") is None
        assert guess_problem_class_from_filename("unknown_problem.json") is None
        assert guess_problem_class_from_filename("") is None

    def test_case_sensitivity(self):
        """Test that problem class detection is case-sensitive."""
        # These should not match because the pattern is case-sensitive
        assert guess_problem_class_from_filename("result_mc_graph.json") is None
        assert guess_problem_class_from_filename("result_mis_graph.json") is None

    def test_mc_takes_precedence_over_mis(self):
        """Test that MC is found first if both patterns exist."""
        # MC appears first in the if-elif chain
        result = guess_problem_class_from_filename("result_MC_MIS_graph.json")
        assert result == "MC"


class TestGuessProblemClass:
    """Tests for guess_problem_class function."""

    def test_from_result_dict_with_problem_class(self):
        """Test extracting problem class from result dictionary."""
        result = {
            "problem_class": "MC:some_init_string"
        }
        with patch("qaoa_parameter_setting.utils.metadata.PROBLEM_CLASSES") as mock_classes:
            mock_class = Mock()
            mock_class.from_str.return_value = "MC"
            mock_classes.__getitem__.return_value = mock_class
            mock_classes.__contains__.return_value = True
            
            problem_class = guess_problem_class("result.json", result)
            assert problem_class == "MC"
            mock_class.from_str.assert_called_once_with("some_init_string")

    def test_from_result_dict_without_init_string(self):
        """Test extracting problem class without initialization string."""
        result = {
            "problem_class": "MIS"
        }
        with patch("qaoa_parameter_setting.utils.metadata.PROBLEM_CLASSES") as mock_classes:
            mock_class = Mock()
            mock_class.from_str.return_value = "MIS"
            mock_classes.__getitem__.return_value = mock_class
            mock_classes.__contains__.return_value = True
            
            problem_class = guess_problem_class("result.json", result)
            assert problem_class == "MIS"
            mock_class.from_str.assert_called_once_with("")

    def test_from_result_dict_unknown_class(self):
        """Test with unknown problem class in result dict."""
        result = {
            "problem_class": "UNKNOWN:init"
        }
        with patch("qaoa_parameter_setting.utils.metadata.PROBLEM_CLASSES") as mock_classes:
            mock_classes.__contains__.return_value = False
            
            # Should fall back to filename
            problem_class = guess_problem_class("result_MC_graph.json", result)
            assert problem_class == "MC"

    def test_from_filename_when_no_result_dict(self):
        """Test extracting problem class from filename when no result dict."""
        problem_class = guess_problem_class("result_MC_graph.json", result=None)
        assert problem_class == "MC"
        
        problem_class = guess_problem_class("result_MIS_graph.json", result=None)
        assert problem_class == "MIS"

    def test_from_filename_when_result_dict_has_no_problem_class(self):
        """Test falling back to filename when result dict has no problem_class."""
        result = {
            "some_other_key": "value"
        }
        problem_class = guess_problem_class("result_MC_graph.json", result)
        assert problem_class == "MC"

    def test_returns_none_when_not_found(self):
        """Test that None is returned when problem class cannot be determined."""
        result = {}
        problem_class = guess_problem_class("unknown_result.json", result)
        assert problem_class is None

    def test_result_dict_takes_precedence_over_filename(self):
        """Test that result dict takes precedence over filename."""
        result = {
            "problem_class": "MIS"
        }
        with patch("qaoa_parameter_setting.utils.metadata.PROBLEM_CLASSES") as mock_classes:
            mock_class = Mock()
            mock_class.from_str.return_value = "MIS"
            mock_classes.__getitem__.return_value = mock_class
            mock_classes.__contains__.return_value = True
            
            # Filename says MC, but result dict says MIS
            problem_class = guess_problem_class("result_MC_graph.json", result)
            assert problem_class == "MIS"

    def test_with_complex_problem_class_string(self):
        """Test with complex problem class initialization string."""
        result = {
            "problem_class": "MC:param1=value1,param2=value2"
        }
        with patch("qaoa_parameter_setting.utils.metadata.PROBLEM_CLASSES") as mock_classes:
            mock_class = Mock()
            mock_class.from_str.return_value = "MC"
            mock_classes.__getitem__.return_value = mock_class
            mock_classes.__contains__.return_value = True
            
            problem_class = guess_problem_class("result.json", result)
            assert problem_class == "MC"
            mock_class.from_str.assert_called_once_with("param1=value1,param2=value2")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_total_runtime_with_missing_train_duration(self):
        """Test total_runtime when some entries don't have train_duration."""
        data = {
            "iteration_0": {"train_duration": 10.0},
            "iteration_1": {},  # Missing train_duration
        }
        # This should raise a KeyError
        with pytest.raises(KeyError):
            total_runtime(data)

    def test_total_runtime_with_negative_durations(self):
        """Test total_runtime with negative durations."""
        data = {
            "iteration_0": {"train_duration": -5.0},
            "iteration_1": {"train_duration": 10.0},
        }
        # Should still calculate, even if negative
        assert total_runtime(data) == pytest.approx(5.0)

    def test_guess_problem_class_with_empty_filename(self):
        """Test guess_problem_class with empty filename."""
        assert guess_problem_class("", result=None) is None

    def test_guess_problem_class_with_none_result(self):
        """Test that None result is handled correctly."""
        problem_class = guess_problem_class("result_MC_graph.json", result=None)
        assert problem_class == "MC"

    def test_filename_with_multiple_problem_class_markers(self):
        """Test filename with both MC and MIS markers."""
        # MC should be found first
        assert guess_problem_class_from_filename("result_MC_MIS_graph.json") == "MC"
        # MIS should be found if MC is not present
        assert guess_problem_class_from_filename("result_MIS_graph.json") == "MIS"

    def test_total_runtime_with_float_precision(self):
        """Test total_runtime maintains float precision."""
        data = {
            "iteration_0": {"train_duration": 0.1},
            "iteration_1": {"train_duration": 0.2},
            "iteration_2": {"train_duration": 0.3},
        }
        result = total_runtime(data)
        assert isinstance(result, (int, float))
        assert result == pytest.approx(0.6)

    def test_guess_problem_class_with_malformed_result_dict(self):
        """Test with malformed result dictionary."""
        result = {
            "problem_class": None
        }
        # Should fall back to filename
        problem_class = guess_problem_class("result_MC_graph.json", result)
        assert problem_class == "MC"