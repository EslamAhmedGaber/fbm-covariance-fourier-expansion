# Code

Layout
- `cfgp/core.py` - features (CovFourier coordinates, residual energies), bounds (L in its diagonal form, the trace form,
  U0/U1/U2 with the per-vector majorant), restricted likelihood, profiles, baselines (exact evidence, inducing points,
  optimised pseudo-inputs, cosine-only features, best rank-D approximation).
- `covfourier/` - unchanged copy of the public CovFourier package, release v1.3.0 (closed-form moments, cosine Gram).
- `experiments/` - one script per experiment of `experiments/PROTOCOL.md`:
  `e1_bracket_exact.py` (E1, E2), `e3_feature_count.py`, `e4_benchmark_rank.py`, `e5_learn_hurst.py`,
  `e5b_enclosure.py`, `e5c_enclosure.py`, `e6_scale.py`, `e7_real_rivers.py`, `e7b_per_basin.py`,
  `e7c_per_basin_centre_train.py`, `e8_real_boreholes.py`, `e10_e11_e12.py` (argument E10, E11 or E12),
  `f1_idea_data.py` (Figure 1 data), `common.py` (designs, seeds, logging).
- `figures/make_figures.py` - Figures 1-5 from the stored results only.

Running
- Python 3.11 with NumPy 2.4, SciPy 1.17, mpmath 1.4, pandas 2.3, Matplotlib 3.11 (psutil, threadpoolctl optional).
- `python experiments/e1_bracket_exact.py P 0.75` etc. (kinds P, W, S; Hurst indices optional); each script writes
  `results/<name>.json` and a log in `logs/`. E3 for windows and heat kernels was run with `E3_MS=1000,10000`;
  E6 needs a scratch folder: `python experiments/e6_scale.py <scratch_dir> 0.75` (about 4 GB of chunk files).
- E7 reads the CAMELS files under `04_DATA`; E8 reads the NOAA profiles under `04_DATA`, the bundled
  `borehole_model/` modules and CRUTEM5 for the site selection (see the top-level README).
- `python figures/make_figures.py` regenerates `figures/Figure_1..5.{pdf,png}` and `figures/figure_checks.json`.
