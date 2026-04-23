"""Utility methods to extract metadata from the result data files."""

from typing import Any

from qaoa_training_pipeline.utils.problem_classes import PROBLEM_CLASSES

from qaoa_parameter_setting.utils.types import ProblemClass


def total_runtime(data: dict):
    """Get the total runtime of the training."""
    exclude = ["args", "pre_processing", "cost_operator"]

    total_duration = 0

    for key, value in data.items():
        if key in exclude:
            continue

        total_duration += value["train_duration"]

    return total_duration


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
