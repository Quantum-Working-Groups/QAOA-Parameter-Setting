#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

DIR1=""
DIR2=""
SAVE_DIR=""
DEPTHS=()

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$DIR1" || -z "$DIR2" || -z "$SAVE_DIR" ]]; then
    echo "Usage: $0 --instance_dir PATH --method_dir PATH --save_dir PATH"
    exit 1
fi

for file1 in "$DIR1"/*; do
  for file2 in "$DIR2"/*; do
    base1=${file1##*/}
    base1=${base1%.json}
    base2=${file2##*/} 
    base2=${base2%.json}
    if [[ "$base2" == *interp* || "$base2" == *transitionstates* ]]; then
      tk_flag="--train_kwargs2"
    else
      tk_flag="--train_kwargs0"
    fi

    for depth in "${DEPTHS[@]}"; do
      echo "Processing: $file1 with $file2 and depth $depth"
      python -m train \
        --input "$file1" \
        --config "$file2" \
        --save \
        --save_dir "$SAVE_DIR" \
        --save_file "__${base1}_${base2}_depth_${depth}.json" \
        --pre_factor -0.5 \
        "$tk_flag" "reps:$depth"
    done
  done
done
