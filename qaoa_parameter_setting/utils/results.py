"""Utility functions for handling/parsing results."""

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .labels import sanitize_trainer_config


def sanitize_energy(energy_val: float | None | Literal["NA"]) -> float | None:
    """Sanitize an energy value from a results dictionary.

    Args:
        energy_val: Energy from a results dictionary, which can be a floating value, None, or ``"NA"``.

    Raises:
        ValueError: If the energy is an unknown string.
        ValueError: If the energy is an unknown type.

    Returns:
        A float value or None, representing the input energy. ``"NA`` is mapped to None.
    """
    if isinstance(energy_val, str):
        if energy_val == "NA":
            return None
        else:
            raise ValueError(f"Unknown energy value: {energy_val!r}")
    elif energy_val is None or isinstance(energy_val, (float, np.floating)):
        return energy_val
    else:
        raise ValueError(f"Unknown energy value type: {type(energy_val)!r}")


def result_contains_noopt(result: dict[str, Any]) -> bool:
    """Returns if the given results dictionary contains a trainer whose zeroth iteration is a _no_opt run.

    This function first checks if the trainer JSON file is a TQA or FA
    config, then it checks if _opt is in the config JSON filename. If both
    are true, it returns True. Otherwise, it returns False.

    Args:
        result: The results dictionary.

    Returns:
        True if the zeroth iteration is a _no_opt run.
    """
    raw_config: str = result["args"]["config"]
    config = sanitize_trainer_config(raw_config)

    parts = Path(config).parts[-1].split(".")[0].split("_")
    no_opt_matches = ["TQA", "FA", "FAAer", "TQAAer"]
    if all(x not in parts for x in no_opt_matches):
        return False
    lower_parts = [p.lower() for p in parts]
    if "opt" not in lower_parts:
        return False
    if "noopt" in lower_parts or "no_opt" in config.lower():
        return False
    return True
