from typing import Literal, cast

from .constants import (
    METHOD_ANGLE_OPT_MARKER_PLACEHOLDER,
    METHOD_CONFIG_TO_LABELS,
    METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER,
    OPT_TO_NO_OPT_MAPPING,
    TRAINER_CONFIG_EQUIVALENT_MAPPINGS,
)
from .instance import sanitize_path
from .types import EvaluationType, MethodConfigJSON, MethodJSON


def format_method_label_to(
    label: str,
    format: Literal["text", "latex", "siunitx"] = "latex",
) -> str:
    """Convert method label placeholders to the specified format.

    Args:
        label: Method label string containing placeholder markers
        format: Output format, either "text" or "latex"

    Returns:
        Formatted label string with placeholders replaced according to format
    """
    if format in ["latex", "siunitx"]:
        return label.replace(METHOD_ANGLE_OPT_MARKER_PLACEHOLDER, "$^\\star$").replace(
            METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER, "$^\\dagger$"
        )
    return label


def method_to_method_label(method: MethodJSON) -> str:
    """Convert method JSON to method label string.

    If the method is not known and no human-readable string is defined in
    :attr:`METHOD_CONFIG_TO_LABELS`, the method identifier is returned as-is.

    Args:
        method: The method identifier string, e.g., ``FA_opt.json``.

    Returns:
        The human-readable method label string.
    """
    return METHOD_CONFIG_TO_LABELS.get(method, method)


def trainer_config_to_method(trainer_config: MethodConfigJSON) -> MethodJSON:
    """Convert a trainer config to an evaluation-independent method string.

    Args:
        trainer_config: The trainer config filename, e.g., ``FA_SV_opt.json``.

    Returns:
        The method identifier without the evaluation method suffix.
    """
    return MethodJSON(
        trainer_config.replace(
            "_{}".format(trainer_config_to_evaluation(trainer_config)), ""
        )
    )


def trainer_config_to_method_label(config: MethodConfigJSON | MethodJSON) -> str:
    """Convert a method/trainer config to a method label.

    Args:
        config: The trainer config filename, e.g., ``FA_SV_opt.json``, or the
            method without the evaluation method, e.g., ``FA_opt.json``.

    Returns:
        The human-readable method name, e.g., ``Fixed Angles*``.
    """
    try:
        # This will fail if config is a MethodJSON, i.e., without an evaluation
        # string.
        return method_to_method_label(
            trainer_config_to_method(config)
        )  # pyright: ignore[reportArgumentType]
    except ValueError:
        # If we get here, config is most likely a MethodJSON.
        return method_to_method_label(config)  # pyright: ignore[reportArgumentType]


def method_uses_aer(method: MethodConfigJSON | MethodJSON) -> bool:
    """Return if the given method uses Qiskit Aer.

    Args:
        method: The trainer config filename or method name.

    Returns:
        Whether the method uses Qiskit Aer.
    """
    return "Aer" in method


def trainer_config_to_evaluation(trainer_config: MethodConfigJSON) -> EvaluationType:
    """Extract the evaluation method abbreviation from a trainer config.

    Args:
        trainer_config: The method configuration filename.

    Returns:
        The evaluation method abbreviation ("PP", "MPS", or "SV").

    Raises:
        ValueError: If the evaluation method cannot be determined.
    """
    if "PP" in trainer_config:
        return "PP"
    elif "MPS" in trainer_config:
        return "MPS"
    elif "SV" in trainer_config:
        return "SV"
    else:
        raise ValueError(f"Unrecognised energy evaluation for method {trainer_config}")


def sanitize_trainer_config(trainer_config: str, is_old: bool) -> MethodConfigJSON:
    """Sanitize a trainer config string by correcting mislabelled variants.

    This function normalizes trainer config strings that may use inconsistent
    naming conventions (e.g., 'angleOpt' instead of 'angle_opt', 'noOpt' instead
    of 'no_opt'). It looks up the input in the
    TRAINER_CONFIG_EQUIVALENT_MAPPINGS dictionary and returns the correct
    version if found, otherwise returns the input as-is.

    Args:
        trainer_config: The trainer config string, which might be mislabelled.
        is_old: a bool indicating whether the config_path string name comes
            from an old version of the QAOA-Parameter-Setting repository and needs
            special reformatting.

    Returns:
        The correct trainer config string wrapped as MethodConfigJSON.

    Examples:
        >>> sanitize_trainer_config("LR_MPS_angleOpt.json")
        MethodConfigJSON("LR_MPS_angle_opt.json")
        >>> sanitize_trainer_config("FA_SV_opt.json")
        MethodConfigJSON("FA_SV_opt.json")
    """
    if is_old:
        return TRAINER_CONFIG_EQUIVALENT_MAPPINGS.get(
            trainer_config, MethodConfigJSON(trainer_config)
        )
    else:
        return trainer_config


def config_path_to_config(config_path: str, is_old: bool) -> MethodConfigJSON:
    """Convert a string path to a config/method to just the config name.

    Note that this function does not validate that the config filename is valid,
    only that it is a valid filename.

    Args:
        config_path: A path to a method config file, to be sanitized and
            converted into a valid :type:`MethodConfigJSON`.
        is_old: a bool indicating whether the config_path string name comes
            from an old version of the QAOA-Parameter-Setting repository and needs
            special reformatting.

    Returns:
        The name of a method config JSON file, without relative or absolute
        paths.
    """
    basename = sanitize_path(config_path).split("/")[-1]
    return sanitize_trainer_config(basename, is_old)


def trainer_config_to_evaluation_label(trainer_config: MethodConfigJSON) -> str:
    """Extract the evaluation method label from a trainer config.

    Args:
        trainer_config: The method configuration filename.

    Returns:
        The evaluation-method label, e.g., ``"MPS (Aer)"`` or ``"SV"``.
    """
    evaluation = trainer_config_to_evaluation(trainer_config)
    uses_aer = method_uses_aer(trainer_config)
    if evaluation == "MPS":
        if uses_aer:
            return "MPS (Aer)"
        return "MPS (Quimb)"
    else:
        if uses_aer:
            raise ValueError(
                f"Trainer config {trainer_config:!r} uses Aer but is not MPS."
            )
        return evaluation


def trainer_config_to_no_opt(trainer_config: MethodConfigJSON) -> MethodConfigJSON:
    """Return the ``no_opt`` variant of the trainer config filename

    Args:
        trainer_config: The method configuration filename, typically ending in
            ``_opt.json``.

    Returns:
        Modified trainer config name.
    """
    noopt_trainer_config: str | None = None
    # Check if any opt method exists. If yes, replace it and break.
    for k in OPT_TO_NO_OPT_MAPPING.keys():
        if k in trainer_config:
            noopt_trainer_config = trainer_config.replace(k, OPT_TO_NO_OPT_MAPPING[k])
            break
    # If we don't have an explicit mapping, just replace 'opt' with 'no_opt'.
    if noopt_trainer_config is None:
        noopt_trainer_config = trainer_config.replace("opt", "no_opt")
    return cast(MethodConfigJSON, noopt_trainer_config)


def method_sort_key(method_label: str) -> str:
    """The sort key for the method label.

    Labels are sorted by their prefix, without the optimised or unoptimised
    markers. Then, within a method, the labels are ordered by angle-optimised,
    then parameter-optimised, then unoptimised. To accomplish this, the
    method_labels have their markers removed and replaced by known suffixes with
    a known ordering.

    Args:
        method_label: A method label generated by :fun:`method_to_method_label`.

    Returns:
        A string-key suitable for sorting method labels.
    """
    SUFFIX_ANGLE_OPT = "_1"
    SUFFIX_NO_OPT = "_3"
    # We need to see if any markers are present. If not, they we mark the
    # standard method as a parameter-optimised method label, which must come
    # second.
    if (
        METHOD_ANGLE_OPT_MARKER_PLACEHOLDER not in method_label
        and METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER not in method_label
    ):
        return method_label + "_2"
    # If we have a marker, then we just replace it.
    return method_label.replace(
        METHOD_ANGLE_OPT_MARKER_PLACEHOLDER, SUFFIX_ANGLE_OPT
    ).replace(METHOD_NO_OPT_AT_ALL_MARKER_PLACEHOLDER, SUFFIX_NO_OPT)
