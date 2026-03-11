"""Class and function to summarise training data into LaTeX tables."""

from .summary_table_formatter import (
    formatted_styler_for,
    format_method_name,
)
from .summary_table import SummaryTable

ACRONYM_MAPPING = {
    "F": "Fourier",
    "FAer": "Fourier (Aer)",
    "FAAer": "Fixed Angle (Aer)",
    "FA": "Fixed Angle",
    "I": "Interp",
    "IAer": "Interp (Aer)",
    "LR": "Linear Ramp",
    "LRAer": "Linear Ramp (Aer)",
    "RTS": "Recursive TS",
    "TS": "Recursive TS",
    "TQA": "TQA",
    "TQAAer": "TQA (Aer)",
    "PT": "Param. Transfer",
}
"""Mapping of method acronyms to full phrases."""

ACRONYM_MAPPING_WITHOUT_AER_SUFFIX = {
    "F": "Fourier",
    "FAer": "Fourier",
    "FAAer": "Fixed Angle",
    "FA": "Fixed Angle",
    "I": "Interp",
    "IAer": "Interp",
    "LR": "Linear Ramp",
    "LRAer": "Linear Ramp",
    "RTS": "Recursive TS",
    "TS": "Recursive TS",
    "TQA": "TQA",
    "TQAAer": "TQA",
    "PT": "Param. Transfer",
}
"""Mapping of method acronyms to full phrases, without ``(Aer)`` suffixes."""
