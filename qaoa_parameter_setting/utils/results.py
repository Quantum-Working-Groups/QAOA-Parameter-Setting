"""Utility functions for handling/parsing results."""

from enum import Enum
from typing import Any, Literal

import numpy as np

from .constants import OPT_TO_NO_OPT_MAPPING
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


class DerivedType(Enum):
    ZEROTH_ITER_IS_NOOPT = 0
    """If the zeroth trainer optimised result is a ``_no_opt`` variant."""
    ZEROTH_ITER_IS_LR_OPT = 1
    """If the zeroth trainer optimised result is a ``LR_*_opt`` variant."""
    INITIAL_PARAMS_LR_NO_OPT = 2
    """If the zeroth trainer initial parameters are a ``LR_*_no_opt`` variant."""


def __config_derived_flags(config: MethodConfigJSON) -> list[DerivedType]:
    config_lower = config.lower()

    # 1. If the zeroth trainer's optimised result is a _no_opt variant of this config.
    is_zeroth_iter_noopt = False
    if any(
        term.lower() in config_lower for term in ["TQA_", "FA_", "TQAAer_", "FAAer_"]
    ):
        if (
            "_opt" in config_lower
            and "_angle_opt" not in config_lower
            and "_no_opt" not in config_lower
        ):
            is_zeroth_iter_noopt = True

    # 2.If the zeroth trainer's optimised result is an LR_*_opt variant of this config.
    is_zeroth_iter_lr_opt = False
    if (
        any(term.lower() in config_lower for term in ["LR_"])
        and "_angle_opt" in config_lower
    ):
        is_zeroth_iter_lr_opt = True

    # 3. If the initial parameters of the zeroth trainer's result is an LR_*_no_opt variant of this config.
    is_inital_param_lr_no_opt = False
    if any(term.lower() in config_lower for term in ["LR_"]):
        is_inital_param_lr_no_opt = True

    # Return all applicable DerivedType flags.
    derived_flags: list[DerivedType] = []
    if is_zeroth_iter_noopt:
        derived_flags.append(DerivedType.ZEROTH_ITER_IS_NOOPT)
    if is_zeroth_iter_lr_opt:
        derived_flags.append(DerivedType.ZEROTH_ITER_IS_LR_OPT)
    if is_inital_param_lr_no_opt:
        derived_flags.append(DerivedType.INITIAL_PARAMS_LR_NO_OPT)
    return derived_flags


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
    derived_types = __config_derived_flags(config)
    return len(derived_types) > 0


def get_derived_configs(config: str) -> list[tuple[str, DerivedType]]:
    """Get the derived config names from an angle_opt or opt config.

    For TQA/FA/TQAAer/FAAer with _opt, returns the no_opt variant.
    For LR with angle_opt, returns the non-angle variant (e.g., LR_MPS_opt.json).
    For LR with either angle_opt or opt, returns the unoptimised initial parameters (e.g., LR_MPS_no_opt.json).

    Args:
        config: The trainer config filename.

    Returns:
        A list of derived config names. If no derived configs exist, the list is empty.
    """

    config = config_path_to_config(config)
    derived_types = __config_derived_flags(config)

    derived_configs: list[tuple[str, DerivedType]] = []

    if len(derived_types) == 0:
        raise ValueError(f"Config {config!r} does not contain a derived method.")

    for derived_type in derived_types:
        if derived_type == DerivedType.ZEROTH_ITER_IS_NOOPT:
            # Use OPT_TO_NO_OPT_MAPPING as the source of truth.
            # FA -> _no_opt suffix; TQA -> bare-flag (strip _opt).
            no_opt = OPT_TO_NO_OPT_MAPPING.get(
                config, config.replace("_opt", "_no_opt")
            )
            derived_configs.append((no_opt, derived_type))
        if derived_type == DerivedType.ZEROTH_ITER_IS_LR_OPT:
            # LR _angle_opt zeroth iter is the _opt variant (parameter-only).
            derived_configs.append(
                (config.replace("_angle_opt", "_opt"), derived_type)
            )
        if derived_type == DerivedType.INITIAL_PARAMS_LR_NO_OPT:
            # LR initial params are the bare-flag (no-opt) variant.
            # Strip _angle_opt -> _opt first, then strip _opt -> bare flag.
            derived_configs.append(
                (
                    config.replace("_angle_opt", "_opt").replace("_opt", ""),
                    derived_type,
                )
            )

    return derived_configs
