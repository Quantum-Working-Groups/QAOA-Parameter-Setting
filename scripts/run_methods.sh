#!/usr/bin/env bash
set -eu
set -o pipefail
shopt -s nullglob

DIR1=""
DIR2=""
SAVE_DIR=""
DEPTHS=()
USE_GPU=false
PROBLEM_CLASS=""


while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance_dir)
            DIR1="$2"
            shift 2
            ;;
        --method_dir)
            DIR2="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
	    --depths)
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                DEPTHS+=("$1")
                shift
            done
	    ;;
        --problem_class)
            PROBLEM_CLASS="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;

        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$DIR2" || -z "$SAVE_DIR" ]]; then
    echo "Usage: $0 --method_dir PATH --save_dir PATH"
    exit 1
fi

# For LABS, create a dummy instance file if none provided
if [[ -n "$PROBLEM_CLASS" && "$PROBLEM_CLASS" == labs* ]]; then
    if [[ -z "$DIR1" ]]; then
        DIR1=$(mktemp -d)
        echo '{"edge list": []}' > "$DIR1/dummy.json"
        trap "rm -rf $DIR1" EXIT
    fi
else
    # For non-LABS problems, instance_dir is required
    if [[ -z "$DIR1" ]]; then
        echo "Error: --instance_dir is required for non-LABS problems"
        echo "Usage: $0 --instance_dir PATH --method_dir PATH --save_dir PATH"
        echo "   or: $0 --method_dir PATH --save_dir PATH --problem_class labs:N"
        exit 1
    fi
fi


for file1 in "$DIR1"/*; do
  for file2 in "$DIR2"/*; do
    base1=${file1##*/}
    base1=${base1%.json}
    base2=${file2##*/} 
    base2=${base2%.json}

    # For LABS problems, use problem class in filename
    if [[ -n "$PROBLEM_CLASS" && "$PROBLEM_CLASS" == labs* ]]; then
        base1=${PROBLEM_CLASS//:/_}
    fi

    # For LABS, replace EfficientDepthOneEvaluator with StatevectorEvaluator
    # and ensure energy_minimization is set to true
    # (LABS has quartic terms which EfficientDepthOneEvaluator cannot handle)
    method_file="$file2"
    if [[ -n "$PROBLEM_CLASS" && "$PROBLEM_CLASS" == labs* ]]; then
        method_file=$(mktemp --suffix=.json)
        sed -e 's/EfficientDepthOneEvaluator/StatevectorEvaluator/g' \
            -e 's|"minimize_args":|"energy_minimization": true,\n                "minimize_args":|g' \
            "$file2" > "$method_file"
    fi

    # Pass 'reps' to the correct trainer in the chain (index 2 for recursive/interp methods)
    tk_flag="--train_kwargs0"
    if grep -Eq '"trainer":\s*"(Recursion)Trainer"' "$method_file"; then
        tk_flag="--train_kwargs2"
    elif [[ "$base2" == *interp* || "$base2" == *transitionstates* ]]; then
        tk_flag="--train_kwargs2"
    fi

    for depth in "${DEPTHS[@]}"; do
        echo "Processing: $file1 with $method_file and depth $depth"

        # Prepare GPU arguments if requested
        gpu_args=""
        if [[ "$USE_GPU" == true && "$file2" == *"SV"* ]]; then
            indices=$(python3 "$(dirname "$0")/get_sv_indices.py" "$method_file")
            for idx in $indices; do
                gpu_args="$gpu_args --evaluator_init_kwargs$idx GPU"
            done
        fi

        cmd=(python -m train \
            --input "$file1" \
            --config "$method_file" \
            --save \
            --save_dir "$SAVE_DIR" \
            --save_file "__${base1}_${base2}_depth_${depth}.json" \
            "$tk_flag" "reps:$depth" \
            $gpu_args
        )
        # Add problem_class if specified, otherwise use pre_factor
        if [[ -n "$PROBLEM_CLASS" ]]; then
            cmd+=(--problem_class "$PROBLEM_CLASS")
        else
            cmd+=(--pre_factor -0.5)
        fi

        "${cmd[@]}"
        done

    # Clean up temp method file if created for LABS
    if [[ "$method_file" != "$file2" ]]; then
        rm -f "$method_file"
    fi
    done
done
