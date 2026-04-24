"""Unit tests for qaoa_parameter_setting/utils/instance.py"""

import pytest
import re
from qaoa_parameter_setting.utils.instance import (
    num_nodes,
    graph_idx,
    num_swap_layers,
    heavy_hex_dimensions,
    regular_degree,
    edge_probability,
    graph_type,
    sanitize_path,
    sanitize_instance_key,
)

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestNumNodes:
    """Tests for num_nodes function."""

    def test_valid_num_nodes(self):
        """Test extracting number of nodes from valid paths."""
        assert num_nodes("path/to/100nodes_graph.json") == 100
        assert num_nodes("10nodes_erdosrenyi.json") == 10
        assert num_nodes("prefix_50nodes_suffix.json") == 50

    def test_num_nodes_with_directory(self):
        """Test extracting number of nodes from paths with directories."""
        assert num_nodes("instances/erdos_renyi/000_11nodes_erdosrenyi50percent.json") == 11
        assert num_nodes("data/100nodes_random3regular.json") == 100

    def test_invalid_num_nodes_raises_error(self):
        """Test that ValueError is raised when pattern not found."""
        with pytest.raises(ValueError, match="Could not determine number of nodes"):
            num_nodes("invalid_path.json")
        with pytest.raises(ValueError, match="Could not determine number of nodes"):
            num_nodes("no_pattern_here.json")


class TestGraphIdx:
    """Tests for graph_idx function."""

    def test_valid_graph_idx(self):
        """Test extracting graph index from valid paths."""
        assert graph_idx("000_11nodes_erdosrenyi.json") == 0
        assert graph_idx("123_graph.json") == 123
        assert graph_idx("999_test.json") == 999

    def test_graph_idx_with_directory(self):
        """Test extracting graph index from paths with directories."""
        assert graph_idx("instances/erdos_renyi/000_11nodes_erdosrenyi50percent.json") == 0
        assert graph_idx("path/to/042_graph.json") == 42

    def test_invalid_graph_idx_raises_error(self):
        """Test that ValueError is raised when pattern not found."""
        with pytest.raises(ValueError, match="Could not determine graph index"):
            graph_idx("no_index_graph.json")
        with pytest.raises(ValueError, match="Could not determine graph index"):
            graph_idx("graph_123.json")  # Index not at beginning


class TestNumSwapLayers:
    """Tests for num_swap_layers function."""

    def test_valid_swap_layers(self):
        """Test extracting number of swap layers from valid paths."""
        assert num_swap_layers("000_100nodes_5swap_layers.json") == 5
        assert num_swap_layers("10swap_layers_graph.json") == 10
        assert num_swap_layers("0swap_layers.json") == 0

    def test_no_swap_layers_returns_none(self):
        """Test that None is returned when pattern not found."""
        assert num_swap_layers("no_swap_pattern.json") is None
        assert num_swap_layers("erdosrenyi_graph.json") is None


class TestHeavyHexDimensions:
    """Tests for heavy_hex_dimensions function."""

    def test_valid_heavy_hex_dimensions(self):
        """Test extracting heavy hex dimensions from valid paths."""
        assert heavy_hex_dimensions("000_1_1_heavyhex_12nodes.json") == (1, 1)
        assert heavy_hex_dimensions("002_7_3_heavyhex_144nodes_weighted.json") == (7, 3)
        assert heavy_hex_dimensions("10_20_heavyhex.json") == (10, 20)

    def test_no_heavy_hex_returns_none(self):
        """Test that None is returned when pattern not found."""
        assert heavy_hex_dimensions("erdosrenyi_graph.json") is None
        assert heavy_hex_dimensions("random_regular.json") is None


class TestRegularDegree:
    """Tests for regular_degree function."""

    def test_valid_regular_degree(self):
        """Test extracting degree from random regular graph paths."""
        assert regular_degree("000_100nodes_random3regular.json") == 3
        assert regular_degree("random6regular_graph.json") == 6
        assert regular_degree("prefix_random9regular_suffix.json") == 9

    def test_no_regular_degree_returns_none(self):
        """Test that None is returned when pattern not found."""
        assert regular_degree("erdosrenyi_graph.json") is None
        assert regular_degree("heavyhex_graph.json") is None


class TestEdgeProbability:
    """Tests for edge_probability function."""

    def test_valid_edge_probability(self):
        """Test extracting edge probability from valid paths."""
        assert edge_probability("000_11nodes_erdosrenyi50percent.json") == 0.5
        assert edge_probability("20percent_graph.json") == 0.2
        assert edge_probability("graph_30percent.json") == 0.3
        assert edge_probability("100percent.json") == 1.0

    def test_no_edge_probability_returns_none(self):
        """Test that None is returned when pattern not found."""
        assert edge_probability("random_regular.json") is None
        assert edge_probability("heavyhex_graph.json") is None


class TestGraphType:
    """Tests for graph_type function."""

    def test_erdos_renyi_type(self):
        """Test identifying Erdos-Renyi graphs."""
        assert graph_type("000_11nodes_erdosrenyi50percent.json") == "erdos_renyi"
        assert graph_type("erdosrenyi_graph.json") == "erdos_renyi"

    def test_random_regular_type(self):
        """Test identifying random regular graphs."""
        assert graph_type("000_100nodes_random3regular.json") == "random_regular"
        assert graph_type("random6regular_graph.json") == "random_regular"

    def test_heavy_hex_type(self):
        """Test identifying heavy hex graphs."""
        assert graph_type("000_1_1_heavyhex_12nodes_weighted.json") == "heavy_hex"
        assert graph_type("7_3_heavyhex.json") == "heavy_hex"

    def test_line_to_full_type(self):
        """Test identifying line to full graphs."""
        assert graph_type("000_100nodes_5swap_layers.json") == "line_to_full"
        assert graph_type("10swap_layers_graph.json") == "line_to_full"

    def test_no_match_raises_error(self):
        """Test that ValueError is raised when no pattern matches."""
        with pytest.raises(ValueError, match="Cannot determine graph type"):
            graph_type("unknown_graph_type.json")

    def test_multiple_matches_raises_error(self):
        """Test that ValueError is raised when multiple patterns match."""
        # This is a contrived example - in practice this shouldn't happen
        # but we test the error handling
        with pytest.raises(ValueError, match="Path matches multiple graph types"):
            graph_type("random3regular_5swap_layers.json")


class TestSanitizePath:
    """Tests for sanitize_path function."""

    def test_windows_path_to_posix(self):
        """Test converting Windows paths to POSIX format."""
        assert sanitize_path("C:\\Users\\test\\file.json") == "C:/Users/test/file.json"
        assert sanitize_path("path\\to\\file.json") == "path/to/file.json"

    def test_posix_path_unchanged(self):
        """Test that POSIX paths remain in POSIX format."""
        assert sanitize_path("path/to/file.json") == "path/to/file.json"
        assert sanitize_path("/home/user/file.json") == "/home/user/file.json"

    def test_no_separators(self):
        """Test paths without separators."""
        assert sanitize_path("file.json") == "file.json"

    def test_mixed_separators(self):
        """Test paths with mixed separators (Windows takes precedence)."""
        result = sanitize_path("path\\to/file.json")
        # Should convert to POSIX
        assert "/" in result
        assert "\\" not in result


class TestSanitizeInstanceKey:
    """Tests for sanitize_instance_key function."""

    def test_sanitize_windows_key(self):
        """Test sanitizing Windows-style instance keys."""
        assert sanitize_instance_key("instances\\erdos_renyi\\graph.json") == "instances/erdos_renyi/graph.json"

    def test_sanitize_posix_key(self):
        """Test sanitizing POSIX-style instance keys."""
        assert sanitize_instance_key("instances/erdos_renyi/graph.json") == "instances/erdos_renyi/graph.json"

    def test_sanitize_simple_key(self):
        """Test sanitizing simple keys without separators."""
        assert sanitize_instance_key("graph.json") == "graph.json"


class TestRegexPatternFor:
    """Tests for the internal __regex_pattern_for function behavior."""

    def test_return_none_instead_false(self):
        """Test that errors are raised when return_none_instead is False."""
        # This is tested indirectly through num_nodes and graph_idx
        with pytest.raises(ValueError):
            num_nodes("invalid.json")

    def test_return_none_instead_true(self):
        """Test that None is returned when return_none_instead is True."""
        # This is tested indirectly through num_swap_layers and heavy_hex_dimensions
        assert num_swap_layers("invalid.json") is None
        assert heavy_hex_dimensions("invalid.json") is None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_values(self):
        """Test handling of zero values."""
        assert num_nodes("0nodes_graph.json") == 0
        assert graph_idx("0_graph.json") == 0
        assert num_swap_layers("0swap_layers.json") == 0
        assert edge_probability("0percent.json") == 0.0

    def test_large_values(self):
        """Test handling of large values."""
        assert num_nodes("999999nodes_graph.json") == 999999
        assert graph_idx("999999_graph.json") == 999999
        assert edge_probability("100percent.json") == 1.0

    def test_multiple_patterns_in_filename(self):
        """Test filenames with multiple matching patterns."""
        # Should extract the first match
        assert num_nodes("10nodes_20nodes_graph.json") == 10
        assert graph_idx("001_002_graph.json") == 1

    def test_case_sensitivity(self):
        """Test that patterns are case-sensitive."""
        # These should not match because patterns are lowercase
        with pytest.raises(ValueError):
            num_nodes("10NODES_graph.json")
        
        # Graph type matching is also case-sensitive
        with pytest.raises(ValueError):
            graph_type("ERDOSRENYI_graph.json")

    def test_special_characters_in_path(self):
        """Test paths with special characters."""
        assert num_nodes("path/with-dashes/100nodes_graph.json") == 100
        assert num_nodes("path_with_underscores/50nodes_graph.json") == 50