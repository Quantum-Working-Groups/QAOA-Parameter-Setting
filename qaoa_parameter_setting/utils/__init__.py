"""Utility subpackages for QAOA parameter-setting analysis.

The package namespace exposes shared helpers for type aliases, plotting and
method labels, graph instance filename parsing, result and metadata handling,
project constants, and Max-Cut graph utilities.
"""

import qaoa_parameter_setting.utils.types as types

import qaoa_parameter_setting.utils.constants as constants
import qaoa_parameter_setting.utils.labels as labels
import qaoa_parameter_setting.utils.instance as instance
import qaoa_parameter_setting.utils.results as results
import qaoa_parameter_setting.utils.metadata as metadata
import qaoa_parameter_setting.utils.graph_utils as problem
