"""Unit tests for qaoa_parameter_setting/utils/labels.py"""

import pytest
from qaoa_parameter_setting.utils.labels import (
    format_method_label_to,
    method_to_method_label,
    trainer_config_to_method,
    trainer_config_to_method_label,
    method_uses_aer,
    trainer_config_to_evaluation,
    trainer_config_to_evaluation_label,
    trainer_config_to_no_opt,
)
from qaoa_parameter_setting.utils.types import MethodConfigJSON, MethodJSON

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestFormatMethodLabelTo:
    """Tests for format_method_label_to function."""

    def test_format_to_latex(self):
        """Test formatting labels to LaTeX format."""
        label = "Fixed Angle*†"
        result = format_method_label_to(label, format="latex")
        assert result == "Fixed Angle$^\\star$$^\\dagger$"

    def test_format_to_siunitx(self):
        """Test formatting labels to siunitx format (same as latex)."""
        label = "Fixed Angle*†"
        result = format_method_label_to(label, format="siunitx")
        assert result == "Fixed Angle$^\\star$$^\\dagger$"

    def test_format_to_text(self):
        """Test formatting labels to text format (no change)."""
        label = "Fixed Angle*†"
        result = format_method_label_to(label, format="text")
        assert result == "Fixed Angle*†"

    def test_format_with_only_angle_opt_marker(self):
        """Test formatting with only angle optimization marker."""
        label = "TQA*"
        result = format_method_label_to(label, format="latex")
        assert result == "TQA$^\\star$"

    def test_format_with_only_no_opt_marker(self):
        """Test formatting with only no optimization marker."""
        label = "Fixed Angle†"
        result = format_method_label_to(label, format="latex")
        assert result == "Fixed Angle$^\\dagger$"

    def test_format_without_markers(self):
        """Test formatting labels without any markers."""
        label = "Linear Ramp"
        result = format_method_label_to(label, format="latex")
        assert result == "Linear Ramp"


class TestMethodToMethodLabel:
    """Tests for method_to_method_label function."""

    def test_known_methods(self):
        """Test converting known method identifiers to labels."""
        assert method_to_method_label(MethodJSON("FA_opt.json")) == "Fixed Angle*"
        assert method_to_method_label(MethodJSON("TQA_opt.json")) == "TQA*"
        assert method_to_method_label(MethodJSON("F.json")) == "Fourier*"
        assert method_to_method_label(MethodJSON("LR_opt.json")) == "Linear Ramp"

    def test_no_opt_methods(self):
        """Test converting no-opt method identifiers to labels."""
        assert method_to_method_label(MethodJSON("FA_no_opt.json")) == "Fixed Angle†"
        assert method_to_method_label(MethodJSON("TQA_no_opt.json")) == "TQA"

    def test_aer_methods(self):
        """Test converting Aer method identifiers to labels."""
        assert method_to_method_label(MethodJSON("FAer.json")) == "Fourier*"
        assert method_to_method_label(MethodJSON("FAAer_opt.json")) == "Fixed Angle*"

    def test_unknown_method_returns_as_is(self):
        """Test that unknown methods return the identifier as-is."""
        unknown = MethodJSON("UnknownMethod.json")
        assert method_to_method_label(unknown) == "UnknownMethod.json"


class TestTrainerConfigToMethod:
    """Tests for trainer_config_to_method function."""

    def test_remove_sv_evaluation(self):
        """Test removing SV evaluation suffix."""
        config = MethodConfigJSON("FA_SV_opt.json")
        result = trainer_config_to_method(config)
        assert result == "FA_opt.json"

    def test_remove_mps_evaluation(self):
        """Test removing MPS evaluation suffix."""
        config = MethodConfigJSON("TQA_MPS_opt.json")
        result = trainer_config_to_method(config)
        assert result == "TQA_opt.json"

    def test_remove_pp_evaluation(self):
        """Test removing PP evaluation suffix."""
        config = MethodConfigJSON("FA_PP_opt.json")
        result = trainer_config_to_method(config)
        assert result == "FA_opt.json"

    def test_with_aer(self):
        """Test removing evaluation from Aer methods."""
        config = MethodConfigJSON("FAAer_MPS_opt.json")
        result = trainer_config_to_method(config)
        assert result == "FAAer_opt.json"


class TestTrainerConfigToMethodLabel:
    """Tests for trainer_config_to_method_label function."""

    def test_with_trainer_config(self):
        """Test converting trainer config to method label."""
        config = MethodConfigJSON("FA_SV_opt.json")
        result = trainer_config_to_method_label(config)
        assert result == "Fixed Angle*"

    def test_with_method_json(self):
        """Test converting method JSON to method label."""
        method = MethodJSON("FA_opt.json")
        result = trainer_config_to_method_label(method)
        assert result == "Fixed Angle*"

    def test_with_no_opt_config(self):
        """Test converting no-opt config to method label."""
        config = MethodConfigJSON("TQA_SV_no_opt.json")
        result = trainer_config_to_method_label(config)
        assert result == "TQA"


class TestMethodUsesAer:
    """Tests for method_uses_aer function."""

    def test_aer_methods(self):
        """Test identifying methods that use Aer."""
        assert method_uses_aer(MethodConfigJSON("FAAer_MPS_opt.json")) is True
        assert method_uses_aer(MethodJSON("FAer.json")) is True
        assert method_uses_aer(MethodConfigJSON("TQAAer_SV_opt.json")) is True

    def test_non_aer_methods(self):
        """Test identifying methods that don't use Aer."""
        assert method_uses_aer(MethodConfigJSON("FA_SV_opt.json")) is False
        assert method_uses_aer(MethodJSON("TQA_opt.json")) is False
        assert method_uses_aer(MethodConfigJSON("F_MPS_opt.json")) is False

    def test_case_sensitivity(self):
        """Test that Aer detection is case-sensitive."""
        assert method_uses_aer(MethodConfigJSON("FAaer_opt.json")) is False
        assert method_uses_aer(MethodConfigJSON("FAAER_opt.json")) is False


class TestTrainerConfigToEvaluation:
    """Tests for trainer_config_to_evaluation function."""

    def test_pp_evaluation(self):
        """Test extracting PP evaluation type."""
        config = MethodConfigJSON("FA_PP_opt.json")
        assert trainer_config_to_evaluation(config) == "PP"

    def test_mps_evaluation(self):
        """Test extracting MPS evaluation type."""
        config = MethodConfigJSON("TQA_MPS_opt.json")
        assert trainer_config_to_evaluation(config) == "MPS"

    def test_sv_evaluation(self):
        """Test extracting SV evaluation type."""
        config = MethodConfigJSON("FA_SV_opt.json")
        assert trainer_config_to_evaluation(config) == "SV"

    def test_with_aer(self):
        """Test extracting evaluation type from Aer methods."""
        config = MethodConfigJSON("FAAer_MPS_opt.json")
        assert trainer_config_to_evaluation(config) == "MPS"

    def test_unrecognized_evaluation_raises_error(self):
        """Test that ValueError is raised for unrecognized evaluation."""
        config = MethodConfigJSON("FA_UNKNOWN_opt.json")
        with pytest.raises(ValueError, match="Unrecognised energy evaluation"):
            trainer_config_to_evaluation(config)


class TestTrainerConfigToEvaluationLabel:
    """Tests for trainer_config_to_evaluation_label function."""

    def test_mps_with_aer(self):
        """Test MPS evaluation label with Aer."""
        config = MethodConfigJSON("FAAer_MPS_opt.json")
        assert trainer_config_to_evaluation_label(config) == "MPS (Aer)"

    def test_mps_without_aer(self):
        """Test MPS evaluation label without Aer (Quimb)."""
        config = MethodConfigJSON("FA_MPS_opt.json")
        assert trainer_config_to_evaluation_label(config) == "MPS (Quimb)"

    def test_sv_evaluation(self):
        """Test SV evaluation label."""
        config = MethodConfigJSON("FA_SV_opt.json")
        assert trainer_config_to_evaluation_label(config) == "SV"

    def test_pp_evaluation(self):
        """Test PP evaluation label."""
        config = MethodConfigJSON("TQA_PP_opt.json")
        assert trainer_config_to_evaluation_label(config) == "PP"

    def test_sv_with_aer_raises_error(self):
        """Test that SV with Aer raises an error."""
        config = MethodConfigJSON("FAAer_SV_opt.json")
        with pytest.raises(ValueError, match="Invalid format specifier"):
            trainer_config_to_evaluation_label(config)

    def test_pp_with_aer_raises_error(self):
        """Test that PP with Aer raises an error."""
        config = MethodConfigJSON("FAAer_PP_opt.json")
        with pytest.raises(ValueError, match="Invalid format specifier"):
            trainer_config_to_evaluation_label(config)


class TestTrainerConfigTono_opt:
    """Tests for trainer_config_to_no_opt function."""

    def test_known_opt_to_no_opt_mappings(self):
        """Test known mappings from opt to no_opt."""
        assert trainer_config_to_no_opt(MethodConfigJSON("FA_MPS_opt.json")) == "FA_MPS_no_opt.json"
        assert trainer_config_to_no_opt(MethodConfigJSON("FA_PP_opt.json")) == "FA_PP_no_opt.json"
        assert trainer_config_to_no_opt(MethodConfigJSON("FA_SV_opt.json")) == "FA_SV_no_opt.json"
        assert trainer_config_to_no_opt(MethodConfigJSON("TQA_MPS_opt.json")) == "TQA_MPS_no_opt.json"
        assert trainer_config_to_no_opt(MethodConfigJSON("TQA_PP_opt.json")) == "TQA_PP_no_opt.json"
        assert trainer_config_to_no_opt(MethodConfigJSON("TQA_SV_opt.json")) == "TQA_SV_no_opt.json"

    def test_aer_opt_to_no_opt_mappings(self):
        """Test Aer method mappings from opt to no_opt."""
        assert trainer_config_to_no_opt(MethodConfigJSON("FA_MPSAer_opt.json")) == "FA_MPSAer_no_opt.json"
        assert trainer_config_to_no_opt(MethodConfigJSON("TQA_MPSAer_opt.json")) == "TQA_MPSAer_no_opt.json"

    def test_generic_opt_to_no_opt(self):
        """Test generic replacement of 'opt' with 'no_opt'."""
        # For methods not in the explicit mapping
        config = MethodConfigJSON("CustomMethod_opt.json")
        result = trainer_config_to_no_opt(config)
        assert result == "CustomMethod_no_opt.json"

    def test_angle_opt_to_no_opt(self):
        """Test conversion of angle_opt to no_opt."""
        config = MethodConfigJSON("LR_angle_opt.json")
        result = trainer_config_to_no_opt(config)
        # Should use generic replacement
        assert "no_opt" in result

    def test_already_no_opt(self):
        """Test that no_opt configs are handled correctly."""
        config = MethodConfigJSON("FA_MPS_no_opt.json")
        result = trainer_config_to_no_opt(config)
        # Should replace 'opt' with 'no_opt', resulting in 'no_no_opt'
        assert "no_no_opt" in result


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        # These should not crash but may return unexpected results
        assert method_to_method_label(MethodJSON("")) == ""
        assert method_uses_aer(MethodJSON("")) is False

    def test_multiple_evaluation_types_in_name(self):
        """Test handling of multiple evaluation type markers."""
        # Should match the first one found
        config = MethodConfigJSON("FA_PP_MPS_opt.json")
        assert trainer_config_to_evaluation(config) == "PP"

    def test_case_variations(self):
        """Test case sensitivity in various functions."""
        # Evaluation detection should be case-sensitive
        with pytest.raises(ValueError):
            trainer_config_to_evaluation(MethodConfigJSON("FA_sv_opt.json"))

    def test_special_characters_in_method_names(self):
        """Test method names with special characters."""
        method = MethodJSON("Method-With-Dashes.json")
        assert method_to_method_label(method) == "Method-With-Dashes.json"

    def test_format_label_with_multiple_markers(self):
        """Test formatting labels with both markers."""
        label = "Method*†"
        latex_result = format_method_label_to(label, format="latex")
        assert "$^\\star$" in latex_result
        assert "$^\\dagger$" in latex_result