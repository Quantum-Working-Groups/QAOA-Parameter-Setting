#!/usr/bin/env python3
"""Enable GPU for StatevectorEvaluator in a training config file."""

import json
import sys
from pathlib import Path


def enable_gpu(method_file_path: Path) -> None:
    """Modify method file to enable GPU for all StatevectorEvaluator instances.

    Args:
        method_file_path: Path to the JSON method file to modify in-place.
    """
    with open(method_file_path, encoding="utf-8") as f:
        method_file = json.load(f)

    for trainer in method_file["trainer_chain"]:
        trainer_init = trainer.get("trainer_init", {})
        # Check both top-level and nested trainer_init for StatevectorEvaluator
        for init in [trainer_init, trainer_init.get("trainer_init", {})]:
            if init.get("evaluator") == "StatevectorEvaluator":
                init.setdefault("evaluator_init", {}).setdefault(
                    "statevector_init_args", {}
                )["device"] = "GPU"

    with open(method_file_path, "w", encoding="utf-8") as f:
        json.dump(method_file, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <method_file_path>", file=sys.stderr)
        sys.exit(1)
    enable_gpu(sys.argv[1])

