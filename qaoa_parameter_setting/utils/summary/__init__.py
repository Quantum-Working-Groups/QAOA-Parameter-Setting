"""Class and function to summarise training data into LaTeX tables."""

from .summary_table import SummaryTable
from .summary_table_formatter import format_method_name, formatted_styler_for

ACRONYM_MAPPING = {
    "F": "Fourier",
    "FAer": "Fourier (Aer)",
    "FAAer": {
        # Optimised, i.e., with _opt.json
        True: "Fixed Angle$^\\star$ (Aer)",
        # Unoptimised, i.e., with _no_opt.json or _noOpt.json.
        # Because FA by itself doesn't optimise, we use a dagger.
        False: "Fixed Angle$^\\dagger$ (Aer)",
    },
    "FA": {
        # Optimised, i.e., with _opt.json
        True: "Fixed Angle$^\\star$",
        # Unoptimised, i.e., with _no_opt.json or _noOpt.json.
        # Because FA by itself doesn't optimise, we use a dagger.
        False: "Fixed Angle$^\\dagger$",
    },
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
    "FAAer": {
        # Optimised, i.e., with _opt.json
        True: "Fixed Angle$^\\star$",
        # Unoptimised, i.e., with _no_opt.json or _noOpt.json.
        # Because FA by itself doesn't optimise, we use a dagger.
        False: "Fixed Angle$^\\dagger$",
    },
    "FA": {
        # Optimised, i.e., with _opt.json
        True: "Fixed Angle$^\\star$",
        # Unoptimised, i.e., with _no_opt.json or _noOpt.json.
        # Because FA by itself doesn't optimise, we use a dagger.
        False: "Fixed Angle$^\\dagger$",
    },
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
