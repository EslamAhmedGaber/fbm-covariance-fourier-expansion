# Changelog

## v1.2.0 — 2026-10-31

Accompanies the final revision of the paper (new title: *CovFourier: a covariance-orthonormal Fourier expansion for fractional Brownian motion with two-sided bounds, stable computation and measurement design*).

- Names the method CovFourier throughout the documentation and the figure legends.
- Adds the worked example of the CovFourier procedure (`code/applications/p1_worked_example.py`, `results/P1_worked_example.json`).
- Adds model selection for the borehole study (`app7_model_selection.py`): the amplitude grid is extended until every optimum is interior; fit and held-out likelihood select `H = 1/2`; the regional model is refitted at `H = 0.75`. The re-log forecast is re-run with these hyperparameters (`app7_relog_forecast.py ms`; the earlier output is kept as `A7_relog_forecast_v110.json`).
- Repeats the re-logging design under `H = 1/2` with exact verification (`app6_borehole_design.py H 0.5 256`, `results/A6_borehole_design_H05.json`).
- Redraws Figure 1 as the six-step procedure; renumbers Figures 4-10 to the order of the revised paper (4 multi-output, 5 KL spectrum, 6 weighted functionals, 7-8 sensors, 9 real boreholes, 10 design).
- Small interface additions: `out_name` for `app7_borehole_regional.main`; options for the design and forecast drivers. No change to `code/covfourier/`.

## v1.1.0 — 2026-10-30

First public release at this GitHub repository, incorporating the new-applications manuscript snapshot.

- Adds posterior-variance and design-gap computations, greedy and Frank–Wolfe design, and independent heat-kernel checks.
- Adds synthetic borehole measurement design and scaling comparisons.
- Adds NOAA borehole reconstruction, CRUTEM5 target-error analysis, JMA station checks, regional and re-logging studies, and documented adverse cases.
- Includes Figures 9–10 and the prior Figures 1–8, stored results and scientific drivers.
- Adds portable data download with source hashes, a portable scaling runner, current dependency pins and public reproduction documentation.
- Release validation: 19 core tests, 5 design tests, 167 inherited numerical-claim checks, 71 new numerical-claim checks, and citation consistency for 37 references.

## v1.0.0 — local predecessor

The original four-application package was prepared locally. It was not previously published at this GitHub address. Its numerical methods and studies are retained; original-run environment notes are preserved separately from the new release environment.
