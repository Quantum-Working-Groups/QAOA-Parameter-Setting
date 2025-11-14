"""Script to fix incorrect TQATrainer JSON entries from qaoa-training-pipeline<=15"""

import json
from sys import argv

from qaoa_training_pipeline.training import TQATrainer

filename = argv[-1]

print("Fixing saved JSON at {!r}".format(filename))

with open(filename, "r") as f:
    data = json.load(f)


dummy_trainer = TQATrainer()
# We iterate over trainers, which we know
for _key in data.keys():
    # Trainer entries have integer keys, converted to strings.
    try:
        _ = int(_key)
    except:
        # If we encounter a data entry that isn't a string-integer, ignore it.
        continue

    # Only deal with TQATrainer steps.
    print("Processing trainer index {!r}".format(_key))
    if data[_key]["trainer"]["trainer_name"] == "TQATrainer":
        # The fix is already applied to TQATrainer from version 16. Skip trainer
        # if we are on version 16 or above.
        if data[_key]["system_info"]["qaoa_training_pipeline_version"] >= 16:
            print(
                "Trainer {!r} is a TQA Trainer but with qaoa-training-pipeline>=16".format(_key)
                + " (the fixed version). Skipping."
            )
            continue

        # Check that we haven't already applied this fix.
        if data[_key]["system_info"].get("tqa_trainer_fix_applied", False):
            print("Trainer {!r} has already been fixed.")
            continue
        print("Trainer {!r} is a TQA Trainer; fixing.".format(_key))

        # NOTE: optimized_params and optimized_qaoa_angles were swapped prior to
        # version 16. So we swap the labels to fix the bug.
        optimized_params = data[_key]["optimized_qaoa_angles"]
        optimized_qaoa_angles = data[_key]["optimized_params"]

        # reps is inferred from the number of angles
        reps = len(optimized_qaoa_angles) // 2

        # We expect a certain number of angles and a single TQA parameter.
        assert 2 * reps == len(optimized_qaoa_angles), "Expected an even number of angles."
        assert len(optimized_params) == 1, "Expected a single parameter."

        # Validate that optimized params result in the expected qaoa angles
        expected_angles = dummy_trainer.qaoa_angles_function(optimized_params, reps=reps)
        if len(expected_angles) != len(optimized_qaoa_angles):
            raise RuntimeError("Length of optimized qaoa angles is not the same as computed ones.")
        if not all(x == y for x, y in zip(expected_angles, optimized_qaoa_angles)):
            raise RuntimeError(
                "Optimized QAOA angles for {} reps and dt={} does not match the computed/expected angles.".format(
                    reps, optimized_params[0]
                )
            )

        # All tests pass, so we fix the bug in this trainer step.
        data[_key]["optimized_qaoa_angles"] = optimized_qaoa_angles
        data[_key]["optimized_params"] = optimized_params
        # Record that we've fixed the TQATrainer results.
        data[_key]["system_info"]["tqa_trainer_fix_applied"] = True


print("Saving updated JSON")
# We parse to a string to ensure we don't overwrite the original file with
# partial JSON in-case of failed parsing.
json_str = json.dumps(data, indent="    ")
with open(filename, "w") as f:
    f.write(json_str)
print("JSON fixed for {!r}".format(filename))
