# Script to identify missing Fixed Angle (FA) configurations in a database
# by comparing required depths against available entries for graph instances
import argparse
from pathlib import Path
from sys import stderr
from typing import cast

from qaoa_parameter_setting.utils.types import MethodConfigJSON

# List of Fixed Angle methods to check for missing configurations
DEFAULT_FA_METHODS = cast(
    list[MethodConfigJSON],
    [
        "FA_MPSAer_no_opt.json",
        "FA_MPSAer_opt.json",
        "FA_MPS_no_opt.json",
        "FA_MPS_opt.json",
        "FA_PP_no_opt.json",
        "FA_PP_opt.json",
        "FA_SV_no_opt.json",
        "FA_SV_opt.json",
    ],
)


def main():
    # Parse command-line arguments for database path, depths, instances, and output options
    parser = argparse.ArgumentParser(
        description="Process failed fixed angles lookups from instance files"
    )
    parser.add_argument(
        "--database", type=Path, required=True, help="Path to a JSON database file"
    )
    parser.add_argument(
        "-t",
        "--tqdm",
        action="store_true",
        help="Use tqdm progress bar (default: False)",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        nargs="+",
        required=True,
        help="Depth parameter (must be positive integers)",
    )
    parser.add_argument(
        "-i",
        "--instances",
        type=Path,
        nargs="+",
        help="One or more JSON instance files to process",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        nargs="+",
        help="One or more Fixed Angle methods to check (overrides DEFAULT_FA_METHODS)",
    )

    output_group = parser.add_mutually_exclusive_group(required=False)
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output filename for the missing Fixed Angle configurations. If not provided, the output is printed to stdout.",
    )
    output_group.add_argument(
        "-a",
        "--amend",
        type=Path,
        help="Path to an existing JSON file to amend with missing Fixed Angle configurations. Existing entries with the same instance, depth, and method will be removed before adding new ones.",
    )

    args = parser.parse_args()

    import json

    import networkx as nx
    import numpy as np
    from qaoa_training_pipeline.utils.graph_utils import load_graph

    import qaoa_parameter_setting.utils as utils

    # Validate depth is positive
    if any(d <= 0 for d in args.depth):
        parser.error("--depth must be positive integers")

    # Validate that database file exists
    if not args.database.exists():
        parser.error(f"Database file not found: {args.database}")

    # Validate that all instance files exist
    for instance_file in args.instances:
        if not instance_file.exists():
            parser.error(f"Instance file not found: {instance_file}")

    # Validate that amend file exists if specified
    if args.amend and not args.amend.exists():
        parser.error(f"Amend file not found: {args.amend}")

    # Load database containing existing FA configurations indexed by average degree
    with open(args.database, "r") as f:
        database = json.load(f)

    # Dictionary to store missing FA configurations: {instance: {method: {depth: reason}}}
    missing_fixedangles: dict[str, dict[str, dict[int, str]]] = {}

    # Use provided methods or default to DEFAULT_FA_METHODS
    fa_methods = args.method if args.method else DEFAULT_FA_METHODS

    # Print configuration information to stderr
    print(f"Depths: {args.depth}", file=stderr)
    print("Methods:", file=stderr, end="")
    for i_method, method in enumerate(fa_methods):
        print(
            f"{' ' * len('Methods:') if i_method > 0 else ''} - {method}", file=stderr
        )
    print(file=stderr)
    if args.amend:
        print(f"Output mode: Amending file {args.amend}", file=stderr)
    elif args.output:
        print(f"Output mode: Writing to file {args.output}", file=stderr)
    else:
        print("Output mode: Printing to stdout", file=stderr)
    print(file=stderr)

    # Process each instance file
    iterator = args.instances
    if args.tqdm:
        from tqdm import tqdm

        iterator = tqdm(args.instances, desc="Processing instances")

    def eprint(*print_args):
        """Print to standard error. Useful for logging."""
        if args.tqdm:
            iterator.set_description(" ".join(print_args))
            iterator.update()
        else:
            print(*print_args, file=stderr)

    for instance_file in iterator:
        # Load graph instance
        instance: nx.Graph = load_graph(str(instance_file))

        # Process the instance data with the database
        # (Add your processing logic here)
        eprint(f"Processing {instance_file}...")

        # START: Compute the average degree of the graph
        # Taken from `qaoa_training_pipeline.training.fixed_angle_conjecture`

        assert isinstance(instance.degree, nx.classes.reportviews.DegreeView)
        avg_degree = np.average([degree for _, degree in instance.degree])

        # Round average degree to nearest integer for database lookup
        degree_key = str(int(np.round(avg_degree)))
        # END

        # Determine which depths are missing from the database for this degree
        if degree_key not in database:
            missing_depths = args.depth
        else:
            missing_depths = []
            for depth in args.depth:
                if str(depth) not in database[degree_key]:
                    missing_depths.append(depth)

        # Record missing FA configurations for each instance, method, and depth
        instance_key = utils.instance.sanitize_instance_key(str(instance_file)).split(
            "/"
        )[-1]

        if missing_depths:
            if instance_key not in missing_fixedangles:
                missing_fixedangles[instance_key] = {}
            for _method in fa_methods:
                if _method not in missing_fixedangles[instance_key]:
                    missing_fixedangles[instance_key][_method] = {}
                for depth in missing_depths:
                    missing_fixedangles[instance_key][_method][depth] = (
                        f"No FA for P={depth} and avg. degree. {degree_key}."
                    )

    # Output results: either amend existing file, write to new file, or print to stdout
    if args.amend:
        # Load existing file
        with args.amend.open("r") as f:
            existing_entries: dict[str, dict[str, dict[str, str]]] = json.load(f)

        # Merge new entries into existing entries
        for _inst, methods in missing_fixedangles.items():
            if _inst not in existing_entries:
                existing_entries[_inst] = {}
            for method, depths in methods.items():
                method_key = utils.instance.sanitize_path(method)
                if method_key not in existing_entries[_inst]:
                    existing_entries[_inst][method_key] = {}
                for depth, reason in depths.items():
                    depth_key = str(depth)
                    existing_entries[_inst][method_key][depth_key] = reason

        # Write back to the amend file
        with args.amend.open("w") as f:
            json.dump(existing_entries, f, indent=2)
    elif args.output:
        with args.output.open("w") as f:
            json.dump(missing_fixedangles, f, indent=2)
    else:
        print(
            json.dumps(missing_fixedangles, indent=2),
        )


if __name__ == "__main__":
    main()
