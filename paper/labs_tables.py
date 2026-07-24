"""Reproduce the LABS tables: sidelobe energy, and normalised merit factor / time-to-solution.

Energies are read straight from the runs.  The merit factor and time-to-solution are recomputed
from the saved angles by statevector, which is the slow part (about half a minute per instance at
n = 21, p = 50).  Writes energy.tex and merit_tts.tex next to this script and prints both tables.
"""

import json
from glob import glob
from pathlib import Path

import numpy as np

from labs_qaoa import labs_energies, energy_mf_tts

DATA = Path(__file__).resolve().parents[1] / "data" / "simulations" / "labs"
HERE = Path(__file__).resolve().parent

# label, folder, result key.  "*" rows read the Scipy-refined angles (key "1"), the plain rows the
# schedule they start from (key "0"); the recursive methods keep everything in one file.
METHODS = [("Fourier", "F", None), ("Interp.", "I", None), ("Trivial", "trivial", None),
           ("RTS", "RTS", "2"), ("TQA", "TQA", "0"), ("TQA*", "TQA", "1"),
           ("LR", "LR", "0"), ("LR*", "LR", "1")]


def best_run(folder, key, n):
    """(energy, angles) the table keeps for a method at instance n.

    F/I/trivial hold the whole depth sweep in one recursive file (depth 50 under key "2"), RTS the
    best recursive state at p = 10 (key "2"), LR/TQA the best of the four initial slopes at p = 50.
    """
    if key is None:
        r = json.load(open(glob(str(DATA / folder / f"*labs_{n}_*"))[0]))["2"]["50"]
    elif key == "2":
        r = json.load(open(glob(str(DATA / folder / f"*labs_{n}_*"))[0]))["2"]
    else:
        runs = (json.load(open(f))[key] for f in glob(str(DATA / folder / f"*labs_{n}_*depth_50.json")))
        r = min(runs, key=lambda r: r["energy"])
    return r["energy"], np.array(r["optimized_qaoa_angles"])


def evaluate(nodes):
    energy, mfnorm, tts = {}, {}, {}
    for n in nodes:
        e = labs_energies(n)
        mf_opt = n * n / (2 * e.min())
        for label, folder, key in METHODS:
            _, angles = best_run(folder, key, n)
            E, mf, t = energy_mf_tts(e, angles)
            energy[label, n], mfnorm[label, n], tts[label, n] = E, mf / mf_opt, t
    return energy, mfnorm, tts


def fmt_tts(t):
    return "{:.1f}".format(t) if t < 100 else ("{:.0f}".format(t) if t < 1e4 else "{:.0e}".format(t))


nodes = list(range(10, 22))
energy, mfnorm, tts = evaluate(nodes)
optimal = {n: int(labs_energies(n).min()) for n in nodes}

# --- Table VI: energy, n = 10..21 ---
print("energy (depth 50; RTS depth 10)")
print("      " + "".join("{:>7}".format(n) for n in nodes))
for label, _, _ in METHODS:
    print("{:<8}".format(label) + "".join("{:>7.1f}".format(energy[label, n]) for n in nodes))
print("{:<8}".format("Optimal") + "".join("{:>7d}".format(optimal[n]) for n in nodes))

with open(HERE / "energy.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "r" * len(nodes) + "}\n\\hline\n")
    f.write(" & " + " & ".join(str(n) for n in nodes) + " \\\\\n\\hline\n")
    for label, _, _ in METHODS:
        f.write(label + " & " + " & ".join("{:.1f}".format(energy[label, n]) for n in nodes) + " \\\\\n")
    f.write("Optimal & " + " & ".join(str(optimal[n]) for n in nodes) + " \\\\\n\\hline\n\\end{tabular}\n")

# --- Table III: normalised merit factor and TTS, n = 14..21 ---
cols = list(range(14, 22))
print("\nnormalised merit factor, time-to-solution (depth 50)")
print("      " + "".join("{:>14}".format(n) for n in cols))
for label, _, _ in METHODS:
    row = ["{:.2f}, {}".format(mfnorm[label, n], fmt_tts(tts[label, n])) for n in cols]
    print("{:<8}".format(label) + "".join("{:>14}".format(c) for c in row))

with open(HERE / "merit_tts.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\hline\n")
    f.write(" & " + " & ".join(str(n) for n in cols) + " \\\\\n\\hline\n")
    for label, _, _ in METHODS:
        cells = ["({:.2f}, {})".format(mfnorm[label, n], fmt_tts(tts[label, n])) for n in cols]
        f.write(label + " & " + " & ".join(cells) + " \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
