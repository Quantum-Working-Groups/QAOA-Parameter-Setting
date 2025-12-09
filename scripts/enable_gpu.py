#!/usr/bin/env python3
"""Enable GPU for StatevectorEvaluator in a training config file."""

import json
import sys
from pathlib import Path


def enable_gpu(config_path: Path) -> None:
    """Modify config file to enable GPU for all StatevectorEvaluator instances.

    Args:
        config_path: Path to the JSON config file to modify in-place.
    """
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    for trainer in config["trainer_chain"]:
        # Check top-level trainer_init
        trainer_init = trainer.get("trainer_init", {})
        if trainer_init.get("evaluator") == "StatevectorEvaluator":
            trainer_init.setdefault("evaluator_init", {}).setdefault(
                "statevector_init_args", {}
            )["device"] = "GPU"

        # Check nested trainer_init (for RecursionTrainer, etc.)
        inner_init = trainer_init.get("trainer_init", {})
        if inner_init.get("evaluator") == "StatevectorEvaluator":
            inner_init.setdefault("evaluator_init", {}).setdefault(
                "statevector_init_args", {}
            )["device"] = "GPU"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_file>", file=sys.stderr)
        sys.exit(1)
    enable_gpu(sys.argv[1])

