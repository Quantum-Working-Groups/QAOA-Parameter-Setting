# LABS figures and tables

Reproduces the LABS results from `data/simulations/labs/`.

- `labs_tables.py` ; energy table and the normalised merit factor / time-to-solution table
  (prints both and writes `energy.tex`, `merit_tts.tex`).
- `labs_landscape.py` ; depth-one energy landscape of the 12-spin instance (`landscape.pdf`).
- `labs_depth_sweep.py` ; energy, merit factor and TTS against depth for each instance
  (`method_comparison_N_{n}.pdf`, n = 10..21; n = 21 is Fig. 21).  

`labs_qaoa.py` holds the shared statevector evaluation. The cost Hamiltonian is diagonal, so the
angles are propagated directly rather than through a circuit; it uses cupy when a GPU is available
and falls back to numpy. Needs `numpy` and `matplotlib`, and a LaTeX installation for the figures.

```
python labs_tables.py
python labs_landscape.py
python labs_depth_sweep.py
```
