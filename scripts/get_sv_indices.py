#!/usr/bin/env python3
"""Get indices of trainers using StatevectorEvaluator."""

import json
import sys
from pathlib import Path


def get_indices(method_path: Path) -> None:
    """Print indices of trainers that use StatevectorEvaluator.

    Args:
        method_path: Path to the JSON method file.
    """
    with open(method_path, encoding="utf-8") as f:
        method_config = json.load(f)

    indices = []
    for idx, trainer in enumerate(method_config.get("trainer_chain", [])):
        trainer_init = trainer.get("trainer_init", {})
        # Check both top-level and nested trainer_init for StatevectorEvaluator
        for init in [trainer_init, trainer_init.get("trainer_init", {})]:
            if init.get("evaluator") == "StatevectorEvaluator":
                indices.append(str(idx))
                break

    print(" ".join(indices))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <method_file>", file=sys.stderr)
        sys.exit(1)
    get_indices(sys.argv[1])
