from qaoa_training_pipeline.utils.problem_classes import PROBLEM_CLASSES
from typing import Literal, NewType, TypeAlias, Any

# *** type aliases to make _data type hints easier.
GraphKey: TypeAlias = str
"""GraphKey instance filename."""
Depth: TypeAlias = int
"""Depth of QAOA."""
ProblemClass: TypeAlias = Literal["MC", "MIS"]
"""Optimization problem class."""

MethodConfigJSON = NewType("MethodConfigJSON", str)
"""JSON filename for the method with the energy evaluation method."""
MethodJSON = NewType("MethodJSON", str)
"""JSON filename without the energy evaluation method."""
MethodAcronym = NewType("MethodAcronym", str)
"""Just the acronym for the base parameter training method."""


def guess_problem_class(
    result_filename: str,
    result: dict[str, Any] | None = None,
) -> ProblemClass | None:
    """Guess the problem class from a filename and optional results dictionary.

    Args:
        result_filename: The name of the result file.
        result: The result dictionary.

    Returns:
        The problem class, or None if it could not be determined.
    """
    problem_class: ProblemClass | None = None
    # First determine the problem class from the result dictionary, if provided.
    if result is not None:
        class_str = result.get("problem_class", None)
        # Code taken from qaoa_training_pipeline/train.py
        if class_str is not None:
            class_info = class_str.split(":")
            class_name = class_info[0].lower()

            class_init_str = ""
            if len(class_info) > 1:
                class_init_str = class_info[1]

            if class_name in PROBLEM_CLASSES:
                problem_class = PROBLEM_CLASSES[class_name].from_str(class_init_str)
    if problem_class is None:
        # If no problem class was determined from the result dictionary, try to
        # determine it from the filename.
        problem_class = guess_problem_class_from_filename(result_filename)
    return problem_class


def guess_problem_class_from_filename(result_filename: str) -> ProblemClass | None:
    """Guess the problem class from a filename.

    Args:
        result_filename: Filename for the results JSON.

    Returns:
        The guessed problem class, or None if no class could be determined.
    """
    if "_MC_" in result_filename:
        return "MC"
    elif "_MIS_" in result_filename:
        return "MIS"
    return None
