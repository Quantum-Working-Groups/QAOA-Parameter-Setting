"""Utility functions for handling/parsing results."""

from typing import Any, Literal

import numpy as np

from .labels import config_path_to_config
from .types import MethodConfigJSON


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


def __config_derived_flags(config: MethodConfigJSON) -> tuple[bool, bool, bool, bool]:
    config_lower = config.lower()
    is_no_opt_acronym = any(
        term.lower() in config_lower for term in ["TQA_", "FA_", "TQAAer_", "FAAer_"]
    )
    is_opt_config = (
        "_opt" in config_lower
        and "_angle_opt" not in config_lower
        and "_no_opt" not in config_lower
    )
    is_angle_opt_acronym = any(term.lower() in config_lower for term in ["LR_"])
    is_angle_opt_config = "_angle_opt" in config_lower
    return (is_no_opt_acronym, is_opt_config, is_angle_opt_acronym, is_angle_opt_config)


def result_contains_derived(result: dict[str, Any]) -> bool:
    """Returns if the given results dictionary contains a trainer whose zeroth
    iteration is a derived run.

    This function checks if the config contains derived methods:
    1. TQA/FA/TQAAer/FAAer with _opt (derives no_opt variant)
    2. LR with angle_opt (derives non-angle_opt variant)

    Args:
        result: The results dictionary.

    Returns:
        True if the zeroth iteration contains a derived run, False otherwise.
    """
    raw_config: str = result["args"]["config"]
    config = config_path_to_config(raw_config)
    is_no_opt_acronym, is_opt_config, is_angle_opt_acronym, is_angle_opt_config = (
        __config_derived_flags(config)
    )

    # We check for two things:
    # 1. Can we derive a _no_opt config from this config.
    # 2. Can we derive an _opt config from this _angle_opt config.

    # Check for no_opt derivation (TQA/FA/TQAAer/FAAer with _opt)
    if is_no_opt_acronym and is_angle_opt_acronym:
        raise ValueError(
            f"Result cannot contain both _no_opt and _(angle_)opt derived methods: config={config!r}."
        )
    if is_no_opt_acronym and is_opt_config:
        return True

    # Check for angle_opt derivation (LR with _angle_opt)
    if is_angle_opt_acronym and is_angle_opt_config:
        return True

    return False


def get_derived_config(config: str) -> str:
    """Get the derived config name from an angle_opt or opt config.

    For TQA/FA/TQAAer/FAAer with _opt, returns the no_opt variant.
    For LR with angle_opt, returns the non-angle variant (e.g., LR_MPS_opt.json).

    Args:
        config: The trainer config filename.

    Returns:
        The derived config name, or None if no derivation applies.
    """

    config = config_path_to_config(config)
    is_no_opt_acronym, is_opt_config, is_angle_opt_acronym, is_angle_opt_config = (
        __config_derived_flags(config)
    )

    if is_no_opt_acronym and is_opt_config:
        return config.replace("_opt", "_no_opt")
    if is_angle_opt_acronym and is_angle_opt_config:
        return config.replace("_angle_opt", "_opt")

    raise ValueError(f"Config {config!r} does not contain a derived method.")
