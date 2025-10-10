
"""Utility methods to extract metadata from the result data files."""

def total_runtime(data: dict):
    """Get the total runtime of the training."""
    exclude = ["args", "pre_processing", "cost_operator"]

    total_duration = 0

    for key, value in data.items():
        if key in exclude:
            continue

        total_duration += value["train_duration"]

    return total_duration
