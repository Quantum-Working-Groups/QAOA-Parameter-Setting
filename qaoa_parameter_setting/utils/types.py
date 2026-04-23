from typing import Literal, NewType, TypeAlias

from qaoa_training_pipeline.utils.problem_classes import PROBLEM_CLASSES

# *** type aliases to make _data type hints easier.
GraphKey: TypeAlias = str
"""GraphKey instance filename."""
GraphType = Literal["erdos_renyi", "random_regular", "line_to_full", "heavy_hex"]
"""Graph types."""
Depth: TypeAlias = int
"""Depth of QAOA."""
ProblemClass: TypeAlias = Literal["MC", "MIS"]
"""Optimization problem class."""
EvaluationType: TypeAlias = Literal["MPS", "SV", "PP"]
"""Evaluation types, either ``"MPS"``, ``"SV"``, or ``"PP"``."""
MethodConfigJSON = NewType("MethodConfigJSON", str)
"""JSON filename for the method with the energy evaluation method."""
MethodJSON = NewType("MethodJSON", str)
"""JSON filename without the energy evaluation method."""
MethodAcronym = NewType("MethodAcronym", str)
"""Just the acronym for the base parameter training method."""
