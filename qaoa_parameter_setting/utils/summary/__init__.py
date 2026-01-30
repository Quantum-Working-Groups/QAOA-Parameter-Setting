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
    "TS": "TS",
    "TQA": "TQA",
    "TQAAer": "TQA (Aer)",
}
"""Mapping of method acronyms to full phrases."""
