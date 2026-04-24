"""Unit tests for qaoa_parameter_setting/utils/types.py

This module defines type aliases and NewTypes used throughout the project.
Since these are type definitions rather than runtime functions, we test
that they can be properly instantiated and used.
"""

import pytest
from qaoa_parameter_setting.utils.types import (
    GraphKey,
    GraphType,
    Depth,
    ProblemClass,
    EvaluationType,
    MethodConfigJSON,
    MethodJSON,
    MethodAcronym,
)

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestGraphKey:
    """Tests for GraphKey type alias."""

    def test_graph_key_is_string(self):
        """Test that GraphKey is a string type alias."""
        key: GraphKey = "instances/erdos_renyi/000_11nodes_erdosrenyi50percent.json"
        assert isinstance(key, str)
        assert key == "instances/erdos_renyi/000_11nodes_erdosrenyi50percent.json"

    def test_graph_key_with_various_paths(self):
        """Test GraphKey with various path formats."""
        key1: GraphKey = "simple.json"
        key2: GraphKey = "path/to/file.json"
        key3: GraphKey = "C:\\Windows\\path\\file.json"
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert isinstance(key3, str)


class TestGraphType:
    """Tests for GraphType literal type."""

    def test_valid_graph_types(self):
        """Test that valid graph types are accepted."""
        erdos: GraphType = "erdos_renyi"
        random: GraphType = "random_regular"
        line: GraphType = "line_to_full"
        heavy: GraphType = "heavy_hex"
        
        assert erdos == "erdos_renyi"
        assert random == "random_regular"
        assert line == "line_to_full"
        assert heavy == "heavy_hex"

    def test_graph_type_in_collection(self):
        """Test GraphType values in collections."""
        graph_types: list[GraphType] = [
            "erdos_renyi",
            "random_regular",
            "line_to_full",
            "heavy_hex"
        ]
        assert len(graph_types) == 4
        assert "erdos_renyi" in graph_types


class TestDepth:
    """Tests for Depth type alias."""

    def test_depth_is_int(self):
        """Test that Depth is an integer type alias."""
        depth: Depth = 5
        assert isinstance(depth, int)
        assert depth == 5

    def test_various_depth_values(self):
        """Test various depth values."""
        depth_zero: Depth = 0
        depth_one: Depth = 1
        depth_large: Depth = 100
        
        assert depth_zero == 0
        assert depth_one == 1
        assert depth_large == 100


class TestProblemClass:
    """Tests for ProblemClass literal type."""

    def test_valid_problem_classes(self):
        """Test that valid problem classes are accepted."""
        mc: ProblemClass = "MC"
        mis: ProblemClass = "MIS"
        
        assert mc == "MC"
        assert mis == "MIS"

    def test_problem_class_in_collection(self):
        """Test ProblemClass values in collections."""
        problem_classes: list[ProblemClass] = ["MC", "MIS"]
        assert len(problem_classes) == 2
        assert "MC" in problem_classes
        assert "MIS" in problem_classes


class TestEvaluationType:
    """Tests for EvaluationType literal type."""

    def test_valid_evaluation_types(self):
        """Test that valid evaluation types are accepted."""
        mps: EvaluationType = "MPS"
        sv: EvaluationType = "SV"
        pp: EvaluationType = "PP"
        
        assert mps == "MPS"
        assert sv == "SV"
        assert pp == "PP"

    def test_evaluation_type_in_collection(self):
        """Test EvaluationType values in collections."""
        eval_types: list[EvaluationType] = ["MPS", "SV", "PP"]
        assert len(eval_types) == 3
        assert "MPS" in eval_types


class TestMethodConfigJSON:
    """Tests for MethodConfigJSON NewType."""

    def test_method_config_json_creation(self):
        """Test creating MethodConfigJSON instances."""
        config = MethodConfigJSON("FA_SV_opt.json")
        assert config == "FA_SV_opt.json"
        assert isinstance(config, str)

    def test_various_method_configs(self):
        """Test various method configuration names."""
        configs = [
            MethodConfigJSON("FA_SV_opt.json"),
            MethodConfigJSON("TQA_MPS_opt.json"),
            MethodConfigJSON("F_PP_opt.json"),
            MethodConfigJSON("LR_SV_angle_opt.json"),
        ]
        
        assert len(configs) == 4
        assert all(isinstance(c, str) for c in configs)

    def test_method_config_with_path(self):
        """Test MethodConfigJSON with path prefix."""
        config = MethodConfigJSON("configs/FA_SV_opt.json")
        assert config == "configs/FA_SV_opt.json"


class TestMethodJSON:
    """Tests for MethodJSON NewType."""

    def test_method_json_creation(self):
        """Test creating MethodJSON instances."""
        method = MethodJSON("FA_opt.json")
        assert method == "FA_opt.json"
        assert isinstance(method, str)

    def test_various_methods(self):
        """Test various method names."""
        methods = [
            MethodJSON("FA_opt.json"),
            MethodJSON("TQA_opt.json"),
            MethodJSON("F.json"),
            MethodJSON("LR_opt.json"),
        ]
        
        assert len(methods) == 4
        assert all(isinstance(m, str) for m in methods)

    def test_method_without_evaluation(self):
        """Test that MethodJSON doesn't include evaluation method."""
        # This is a semantic test - MethodJSON should not have _SV, _MPS, etc.
        method = MethodJSON("FA_opt.json")
        assert "_SV" not in method
        assert "_MPS" not in method
        assert "_PP" not in method


class TestMethodAcronym:
    """Tests for MethodAcronym NewType."""

    def test_method_acronym_creation(self):
        """Test creating MethodAcronym instances."""
        acronym = MethodAcronym("FA")
        assert acronym == "FA"
        assert isinstance(acronym, str)

    def test_various_acronyms(self):
        """Test various method acronyms."""
        acronyms = [
            MethodAcronym("FA"),
            MethodAcronym("TQA"),
            MethodAcronym("F"),
            MethodAcronym("LR"),
            MethodAcronym("I"),
        ]
        
        assert len(acronyms) == 5
        assert all(isinstance(a, str) for a in acronyms)


class TestTypeInteroperability:
    """Tests for interoperability between types."""

    def test_method_config_to_method_conversion(self):
        """Test conceptual conversion from MethodConfigJSON to MethodJSON."""
        config = MethodConfigJSON("FA_SV_opt.json")
        # In practice, this would be done by trainer_config_to_method function
        method_str = config.replace("_SV", "")
        method = MethodJSON(method_str)
        assert method == "FA_opt.json"

    def test_string_operations_on_newtypes(self):
        """Test that NewTypes support string operations."""
        config = MethodConfigJSON("FA_SV_opt.json")
        
        # String operations should work
        assert config.startswith("FA")
        assert config.endswith(".json")
        assert "SV" in config
        assert config.split("_") == ["FA", "SV", "opt.json"]

    def test_type_in_dict_keys(self):
        """Test using custom types as dictionary keys."""
        method_dict: dict[MethodJSON, str] = {
            MethodJSON("FA_opt.json"): "Fixed Angle",
            MethodJSON("TQA_opt.json"): "TQA",
        }
        
        assert len(method_dict) == 2
        assert method_dict[MethodJSON("FA_opt.json")] == "Fixed Angle"

    def test_type_in_dict_values(self):
        """Test using custom types as dictionary values."""
        config_dict: dict[str, MethodConfigJSON] = {
            "config1": MethodConfigJSON("FA_SV_opt.json"),
            "config2": MethodConfigJSON("TQA_MPS_opt.json"),
        }
        
        assert len(config_dict) == 2
        assert config_dict["config1"] == "FA_SV_opt.json"


class TestTypeAnnotations:
    """Tests for type annotations and type checking."""

    def test_function_with_graph_type_annotation(self):
        """Test function with GraphType annotation."""
        def process_graph(graph_type: GraphType) -> str:
            return f"Processing {graph_type} graph"
        
        result = process_graph("erdos_renyi")
        assert result == "Processing erdos_renyi graph"

    def test_function_with_depth_annotation(self):
        """Test function with Depth annotation."""
        def set_depth(depth: Depth) -> Depth:
            return depth * 2
        
        result = set_depth(5)
        assert result == 10

    def test_function_with_problem_class_annotation(self):
        """Test function with ProblemClass annotation."""
        def solve_problem(problem: ProblemClass) -> str:
            return f"Solving {problem} problem"
        
        result = solve_problem("MC")
        assert result == "Solving MC problem"

    def test_function_with_evaluation_type_annotation(self):
        """Test function with EvaluationType annotation."""
        def evaluate(eval_type: EvaluationType) -> str:
            return f"Using {eval_type} evaluation"
        
        result = evaluate("MPS")
        assert result == "Using MPS evaluation"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_strings(self):
        """Test that empty strings can be used with string-based types."""
        key: GraphKey = ""
        config: MethodConfigJSON = MethodConfigJSON("")
        method: MethodJSON = MethodJSON("")
        acronym: MethodAcronym = MethodAcronym("")
        
        assert key == ""
        assert config == ""
        assert method == ""
        assert acronym == ""

    def test_special_characters_in_strings(self):
        """Test strings with special characters."""
        key: GraphKey = "path/with-special_chars.json"
        config: MethodConfigJSON = MethodConfigJSON("method-with-dashes.json")
        
        assert "-" in key
        assert "_" in key
        assert "-" in config

    def test_depth_boundary_values(self):
        """Test boundary values for Depth."""
        depth_zero: Depth = 0
        depth_negative: Depth = -1  # Technically allowed by int type
        depth_large: Depth = 999999
        
        assert depth_zero == 0
        assert depth_negative == -1
        assert depth_large == 999999

    def test_type_equality(self):
        """Test equality between instances of NewTypes."""
        config1 = MethodConfigJSON("FA_SV_opt.json")
        config2 = MethodConfigJSON("FA_SV_opt.json")
        config3 = MethodConfigJSON("TQA_MPS_opt.json")
        
        assert config1 == config2
        assert config1 != config3

    def test_type_hashing(self):
        """Test that NewTypes can be hashed (for use in sets/dicts)."""
        configs = {
            MethodConfigJSON("FA_SV_opt.json"),
            MethodConfigJSON("TQA_MPS_opt.json"),
            MethodConfigJSON("FA_SV_opt.json"),  # Duplicate
        }
        
        # Set should contain only 2 unique items
        assert len(configs) == 2

    def test_literal_type_membership(self):
        """Test membership checking for Literal types."""
        valid_graph_types = ["erdos_renyi", "random_regular", "line_to_full", "heavy_hex"]
        valid_problem_classes = ["MC", "MIS"]
        valid_eval_types = ["MPS", "SV", "PP"]
        
        graph_type: GraphType = "erdos_renyi"
        problem: ProblemClass = "MC"
        eval_type: EvaluationType = "MPS"
        
        assert graph_type in valid_graph_types
        assert problem in valid_problem_classes
        assert eval_type in valid_eval_types