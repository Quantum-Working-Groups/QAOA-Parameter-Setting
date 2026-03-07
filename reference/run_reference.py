"""
Solve the MAXCUT SDP relaxation and run random hyperplane roundings for every
graph instance under ../instances/, storing results in ../data/reference/ with
the same subdirectory/filename structure.

Usage (from the repo root or from reference/):
    python reference/run_reference.py [options]

Options:
    --overwrite       Re-process instances whose output file already exists.
    --subdir NAME     Process only the named subdirectory (e.g. erdos_renyi).
    --dry-run         List instances without processing them.

Output JSON schema (one file per instance):
{
    "instance"          : "instances/<type>/<name>.json",
    "n_nodes"           : <int>,
    "n_edges"           : <int>,
    "sdp_value"         : <float>,   # SDP upper bound on MAXCUT
    "sdp_time_seconds"  : <float>,
    "sdp_status"        : <str>,
    "max_cut"           : <float> | null,  # exact MAXCUT from summary_tables_MC.json (if available)
    "roundings": {
        "10":    {"best_cut_value": <float>, "best_cut": [...],
                  "rounding_time_seconds": <float>,
                  "approximation_ratio": <float> | null},  # best_cut_value / max_cut
        "100":   {...},
        "1000":  {...},
        "10000": {...}
    }
}

best_cut is a list of length n_nodes with values +1 or -1 (index = node id).
approximation_ratio is omitted when max_cut is not available in the summary.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Support running from either the repo root or the reference/ directory.
REFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = REFERENCE_DIR.parent
sys.path.insert(0, str(REFERENCE_DIR))

from maxcut_sdp import (
    build_laplacian,
    factorize_sdp_solution,
    run_roundings,
    solve_maxcut_sdp,
)

INSTANCES_DIR = REPO_ROOT / "instances"
OUTPUT_DIR = REPO_ROOT / "data" / "reference"
SUMMARY_PATH = REPO_ROOT / "summary" / "summary_tables_MC.json"
MINMAX_DIR = REPO_ROOT / "data" / "minmax_cuts"
CHECKPOINTS = (10, 100, 1000, 10000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_max_cut_table(summary_path: Path, minmax_dir: Path) -> dict:
    """
    Build a mapping from instance filename -> max_cut value (float).

    Two sources are merged; data/minmax_cuts takes precedence over
    summary_tables_MC.json when both cover the same instance.

    summary_tables_MC.json keys are already plain filenames.
    data/minmax_cuts/<subdir>/<name>_maxmin_cut.json files use a
    '_maxmin_cut' suffix that is stripped to recover the instance filename.
    """
    table = {}

    # Source 1: summary_tables_MC.json (minmax_data section).
    if summary_path.exists():
        print(f"Loading max_cut from {summary_path} ...", end="", flush=True)
        with open(summary_path) as f:
            data = json.load(f)
        for filename, entry in data["minmax_data"].items():
            if entry.get("max_cut") is not None:
                table[filename] = float(entry["max_cut"])
        print(f" {len(table)} entries.", flush=True)
    else:
        print(f"Warning: {summary_path} not found.", flush=True)

    # Source 2: data/minmax_cuts/<subdir>/<name>_maxmin_cut.json.
    # These override summary_tables_MC.json when both are present.
    n_from_dir = 0
    if minmax_dir.exists():
        for subdir in sorted(minmax_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for fpath in sorted(subdir.glob("*_maxmin_cut.json")):
                # Strip '_maxmin_cut' suffix to get the instance filename.
                instance_name = fpath.name.replace("_maxmin_cut.json", ".json")
                with open(fpath) as f:
                    entry = json.load(f)
                if entry.get("max_cut") is not None:
                    table[instance_name] = float(entry["max_cut"])
                    n_from_dir += 1
        print(f"Loaded/updated {n_from_dir} entries from {minmax_dir}.", flush=True)
    else:
        print(f"Warning: {minmax_dir} not found.", flush=True)

    print(f"max_cut table: {len(table)} entries total.", flush=True)
    return table


def load_instance(filepath: Path):
    """Return (n_nodes, edges) from an instance JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    edges = [
        (e["nodes"][0], e["nodes"][1], float(e["weight"]))
        for e in data["edge list"]
    ]
    nodes = {u for u, v, _ in edges} | {v for u, v, _ in edges}
    n = max(nodes) + 1  # nodes are 0-indexed
    return n, edges


def process_instance(filepath: Path, max_cut_table: dict):
    n, edges = load_instance(filepath)
    L = build_laplacian(n, edges)

    print(f"  n={n}, |E|={len(edges)} — solving SDP ...", end="", flush=True)
    X, sdp_value, sdp_time, status = solve_maxcut_sdp(n, L)
    print(
        f" {sdp_time:.2f}s  status={status}  sdp={sdp_value:.6g}",
        flush=True,
    )

    if X is None:
        print(f"  ERROR: SDP returned no solution (status={status})", flush=True)
        return None

    V = factorize_sdp_solution(X)

    print(
        f"  running {CHECKPOINTS[-1]} roundings (checkpoints {CHECKPOINTS}) ...",
        end="",
        flush=True,
    )
    rounding_results = run_roundings(V, edges, checkpoints=CHECKPOINTS)
    t_total = rounding_results[CHECKPOINTS[-1]]["rounding_time_seconds"]
    best_at_max = rounding_results[CHECKPOINTS[-1]]["best_cut_value"]
    print(f" {t_total:.2f}s  best={best_at_max:.6g}", flush=True)

    # Look up exact max_cut by filename (key has no subdirectory prefix).
    max_cut = max_cut_table.get(filepath.name)

    # Annotate each rounding checkpoint with approximation_ratio when available.
    roundings_out = {}
    for k, v in rounding_results.items():
        entry = dict(v)
        if max_cut is not None and max_cut > 0:
            entry["approximation_ratio"] = entry["best_cut_value"] / max_cut
        roundings_out[str(k)] = entry

    rel_path = filepath.resolve().relative_to(REPO_ROOT)

    result = {
        "instance": str(rel_path),
        "n_nodes": n,
        "n_edges": len(edges),
        "sdp_value": float(sdp_value),
        "sdp_time_seconds": float(sdp_time),
        "sdp_status": status,
        "roundings": roundings_out,
    }
    if max_cut is not None:
        result["max_cut"] = float(max_cut)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Re-process already-completed instances.")
    parser.add_argument("--subdir", type=str, default=None, help="Process only this subdirectory.")
    parser.add_argument("--dry-run", action="store_true", help="List instances without processing.")
    args = parser.parse_args()

    subdirs = sorted(d for d in INSTANCES_DIR.iterdir() if d.is_dir())
    if args.subdir:
        subdirs = [d for d in subdirs if d.name == args.subdir]
        if not subdirs:
            sys.exit(f"No subdirectory named '{args.subdir}' found in {INSTANCES_DIR}")

    max_cut_table = load_max_cut_table(SUMMARY_PATH, MINMAX_DIR)

    total_instances = sum(len(list(d.glob("*.json"))) for d in subdirs)
    print(f"Found {total_instances} instances across {len(subdirs)} subdirectories.")
    if args.dry_run:
        for subdir in subdirs:
            instances = sorted(subdir.glob("*.json"))
            covered = sum(1 for f in instances if f.name in max_cut_table)
            print(f"\n  {subdir.name}/  ({len(instances)} files, {covered} with max_cut)")
            for f in instances:
                tag = " [max_cut]" if f.name in max_cut_table else ""
                print(f"    {f.name}{tag}")
        return

    for subdir in subdirs:
        out_subdir = OUTPUT_DIR / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        instances = sorted(subdir.glob("*.json"))
        print(f"\n=== {subdir.name} ({len(instances)} instances) ===")

        for instance_file in instances:
            # Skip if any dated output file already exists for this instance.
            existing = list(out_subdir.glob(f"*_{instance_file.name}"))
            if existing and not args.overwrite:
                print(f"  [skip] {instance_file.name}")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = out_subdir / f"{timestamp}_{instance_file.name}"

            print(f"[{instance_file.name}]")
            result = process_instance(instance_file, max_cut_table)

            if result is not None:
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
