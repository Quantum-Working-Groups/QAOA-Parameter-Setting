"""Function to compute the min- and max-cuts, and sum of edge weights for a
   given graph instance.

Files are saved into an output directory, and are used by
`qaoa_parameter_setting.utils.summary.SummaryTable` to compute the approximation
ratio.
"""

import argparse

parser = argparse.ArgumentParser(
    __file__,
    description="Script to compute the max- and min-cuts for a graph.",
)

_ = parser.add_argument(
    "-w",
    "--overwrite",
    action="store_true",
    help="Whether to overwrite the output file if it already exists.",
)
_ = parser.add_argument(
    "-p",
    "--make-dir",
    action="store_true",
    help="If the output directory doesn't exist, create it.",
)
_ = parser.add_argument(
    "input",
    type=str,
    help="Input graph file.",
)
output_parser = parser.add_mutually_exclusive_group(required=True)
_ = output_parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Optional output file. If not provided, --dir must be provided.",
)
_ = output_parser.add_argument(
    "-d",
    "--dir",
    type=str,
    help="Optional output directory. Output files will have the same name as "
    + "the input file, with the suffix '_maxmin'. If not provided, --output "
    + "must be provided.",
)
_ = output_parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    default=False,
    help="Suppress terminal logging.",
)

args = parser.parse_args()


def log(*pos_args, **kwargs):
    if not args.quiet:
        print(*pos_args, **kwargs)


# ====================================================
# Process input arguments to determine output filename
# ====================================================

from pathlib import Path

# Convert the input into a path for better cross-platform handling.
input_file = Path(args.input)

if args.output is not None and args.dir is not None:
    raise ValueError("Only one of `output` and `dir` can be provided.")

# If we have an output argument, that is the output path.
if args.output is not None:
    output_file = Path(args.output)
else:
    # At this point we only check if dir is None as output is None if we get to this line.
    if args.dir is None:
        raise ValueError("Either `output` or `dir` must be provided.")

    # Filename without extension is the stem.
    _basename = input_file.stem

    # Extension
    _extension = input_file.suffix
    _output_filename = Path(
        "{basename}_maxmin_cut{extension}".format(
            basename=_basename, extension=_extension
        )
    )

    # Join the output filename to the output directory.
    output_file = Path(args.dir).joinpath(_output_filename)

log("Processing {!r} and outputting to {!r}".format(str(input_file), str(output_file)))
# Check if the output file already exists and raise an error if overwrite is
# False.
if output_file.exists() and not args.overwrite:
    raise RuntimeError("Output file already exists but --overwrite not passed.")
_output_dir = output_file.parent
if not _output_dir.exists():
    if args.make_dir:
        # Make the parent directory
        _output_dir.mkdir(parents=True)
    else:
        # Parent directory doesn't exist and we aren't creating the directory.
        # Raise an error.
        raise RuntimeError(
            "Output directory {!r} doesn't exist but ".format(str(_output_dir))
            + "--make-dir not passed.",
        )

# ==============
# Load the graph
# ==============

from qaoa_training_pipeline.utils.graph_utils import (
    load_graph,
    solve_max_cut,
    graph_to_operator,
    operator_to_graph,
)
from time import time_ns

log("Loading graph..")
graph = load_graph(input_file)
log("Convert graph to cost operator...")
# Use a prefactor of -0.5 to compute the min-max cut, as per convention.
cost_op = graph_to_operator(graph, pre_factor=-0.5)
log("Solving for max and min cuts...")
_time_start = time_ns()
max_cut, min_cut, _ = solve_max_cut(cost_op)
_time_end = time_ns()

log("Computing sum-of-weights for easier approximation ratio calculation...")
graph = operator_to_graph(cost_op, pre_factor=-2)
sum_weights = sum(val[2].get("weight", 1.0) for val in graph.edges(data=True))


import json

log("Writing max and min to file.")
with open(output_file, "w") as f:
    json.dump(
        {
            "instance": str(input_file),
            "max_cut": max_cut,
            "min_cut": min_cut,
            "sum_of_weights": sum_weights,
            "time_solve_max_cut_ns": _time_end - _time_start,
        },
        f,
    )
log("Done.")
