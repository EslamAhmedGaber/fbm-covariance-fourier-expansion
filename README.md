# CovFourier: covariance-orthonormal Fourier expansion for fractional Brownian motion

Research software accompanying *CovFourier: shared Gaussian coordinates with exact residual energies for fractional Brownian motion*, by Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman and Sroor M. Elnady.

**v1.3.0 — 6 November 2026.** It accompanies the repositioned paper: the covariance ordering and the law of quadratic functionals (distribution functions, quantiles and tail probabilities with two-sided bounds, checked against the exact law at `H = 1/2`, the exact law of the discretized functional and Monte Carlo), a numerical check of the proof of the closed cosine Gram matrix, the condition number of the scaled computational core, and figures redrawn with one visual language. Section, figure and table numbers follow the repositioned paper. It is a research-software release, not a claim of journal acceptance. v1.1.0 and v1.2.0 remain available unchanged.

## One representation, several tasks

CovFourier is a covariance-orthonormal Fourier expansion. For fractional Brownian motion with `1/2 <= H < 1`, it orthonormalizes a fixed nested family of cosines and two polynomial boundary modes in the covariance inner product. Each admissible deterministic integrand is mapped to shared white coordinates by the same coefficient rule. The coordinates support Wiener-integral simulation, covariance calculations, Gaussian conditioning and measurement design; residual energies quantify the projection error. Together, coordinates and residual energies place every covariance matrix of Wiener integrals between `G G^T` and `G G^T + R` (`G` the stacked coordinates, `R` positive semidefinite with the residual energies on its diagonal), so covariance entries, spectra, laws of quadratic functionals, posterior variances and design objectives receive two-sided brackets.

The mathematical construction and its finite-precision implementation are distinguished throughout: arbitrary black-box samples cannot by themselves certify an unresolved function. Reported numerical certificates depend on resolved Fourier coefficients and the stated numerical-error allowances. At `H = 1/2`, the covariance norm becomes the ordinary `L2` norm; omitting the enrichment recovers the classical cosine system.

## The CovFourier procedure in code

The six steps of Algorithm 1 of the paper map to the package as follows.

| Step | In code |
|---|---|
| 1. Express observations and targets as Wiener integrals | Python callables `f(t)` on `[0, T]`: an indicator for a point value, a window integrand, a filter kernel, a heat kernel |
| 2. Specify `H`, `T`, the dimension `D = n + 2`, the auxiliary length `N` and the input class | arguments of the two constructors below |
| 3. Prepare once | `cf = CovarianceFourier(H, n, T=T, Naux=N)`; `bp = BatchProjector(cf, N=N)` |
| 4. Coordinates of every input by the same rule | `r = bp.from_callables([f1, f2, ...])`; `r["coords"]` has shape `(D, q)` |
| 5. Assess and refine | `r["error2"]`, `r["eta"]`, `r["resolved"]`; repeat with `BatchProjector(cf, N=N // 2)`; increase `n` while a residual exceeds the request |
| 6. Use the shared coordinates | covariances `coords.T @ coords`, joint samples, posterior laws in `D` dimensions, `covfourier.design.InfoDesign` for design brackets, greedy designs and Frank-Wolfe bounds |

The worked example of Section 3 of the paper runs all six steps: `python code/applications/p1_worked_example.py` (results in `results/P1_worked_example.json`).

## Results included

| Task | Evidence in this release |
|---|---|
| Worked example | Four observation types and one target through the six steps; the exact posterior variance (independent quadrature) inside the computed bracket for every `D` from 18 to 514 |
| Stable expansion | Double-precision, matrix-free studies up to `D = 32,769`; comparisons with independent references |
| Spectra and quadratic functionals | KL eigenvalue studies, weighted functionals including `w(t) = 1/t`, deterministic truncation brackets and Monte Carlo comparisons |
| Laws of quadratic functionals | Two-sided brackets for distribution functions and quantiles of `int w B_H^2` (`w = 1` and `w = 1/t`); the exact quantiles at `H = 1/2` lie inside the brackets, extrapolated discretized-law quantiles inside in 24 of 25 cases (the last 3.7e-9 below, within the extrapolation uncertainty), Monte Carlo (10^6 paths) consistent in 30 of 30; two inversion routes agree to 3.3e-11. See [docs/LAWS.md](docs/LAWS.md) |
| Noisy sensors | Experiments up to 128,000 sensors, measured peak RAM 0.68 GiB; a dense float64 covariance alone would occupy about 122 GiB |
| Multiple responses | Smooth and stiff kernels, including cases where specialized quadrature is preferable |
| Measurement design | A posterior-variance bracket and a computable bound on design suboptimality; 21,497-candidate synthetic heat experiment, with reported relative gaps 0.021–0.039% under the stated model (`H = 0.75`) and 0.018–0.066% under the `H = 1/2` selected by the real data, each with numerical allowances |
| Real borehole data | 690 selected profiles; model selection (fit, held-out profiles, independent target) over `H`; regional reconstruction, external temperature comparisons and adverse results retained |

These numbers describe particular stored experiments, not universal speedups or accuracy guarantees. On the real profiles, both the fit and the held-out likelihood select `H = 1/2` among the tested values; prediction of the independent instrumental target does not separate the Hurst indices. After target-error adjustment, nominal 90% site intervals cover about 80%; Japan remains discrepant. The real-network design experiment does not produce a useful tight certificate. See [limitations](docs/LIMITATIONS.md).

## Install and quick start

Python 3.12 was used for the new studies and the release checks. From this repository root:

```sh
python -m pip install -r requirements-tested.txt
python code/tests/test_covfourier.py
python code/tests/test_design.py
```

```python
import sys
sys.path.insert(0, "code")
import numpy as np
from covfourier.engine import CovarianceFourier

cf = CovarianceFourier(H=0.75, n=255)  # build once; D = 257
result = cf.from_callable(lambda t: np.sin(140 * np.pi * t))
print(result["error2"] / result["norm2_aux"])
z = np.random.default_rng(0).standard_normal(len(result["coords"]))
print(result["coords"] @ z)
```

## Reproduction

- [Original studies, four applications and Figures 1–8](docs/ORIGINAL_APPLICATIONS.md).
- [Worked example, real-data and design applications, Figures 9–10](docs/NEW_APPLICATIONS.md).
- [Law of quadratic functionals, Figure 6, Table 4 and Supplementary Table S4 (v1.3.0)](docs/LAWS.md).
- [External data download and attribution](04_DATA/README.md).
- [Checks performed for this release](docs/RELEASE_VALIDATION.md).

Stored result files are included. Large regenerable caches and external raw datasets are excluded. Download the exact external snapshots, checking their SHA-256 hashes, with:

```sh
python tools/fetch_data.py --dataset all
```

For Figures 2–10 and Supplementary Figures S1–S2 from stored results:

```sh
python figures/src/make_figures.py
python figures/src/make_figures_design.py fig10
python figures/src/make_figure_realdata.py
```

LaTeX typography uses TeX Live (`latex`, `dvipng`, Latin Modern); `FIGURES_USETEX=0` selects matplotlib mathtext for scripts supporting that option. The optional Figure 1 preview/build helper needs `requirements-figures.txt` and `pdflatex`. See the original reproduction guide for details. The full scaling batch is available as `python tools/run_design_scaling.py`; it is computationally substantial and appends to its result file.

## Repository contents

- `code/covfourier/`: covariance geometry, projection, whitening, heat kernels and design algorithms.
- `code/tests/`: 19 core regression tests and 5 design tests.
- `code/studies/`, `code/applications/`: experiment drivers.
- `results/`: stored numerical evidence, including adverse and exploratory results.
- `figures/`: figures, plotted data and source scripts.
- `04_DATA/`: source manifests; downloaded raw data are ignored by Git.
- `tools/`: portable download and batch helpers.

The manuscript draft and private working notes are not distributed here. Scientific implementation files are copied unchanged from the completed new-applications workspace; packaging adds portable data retrieval, documentation and release metadata.

## Citation and reuse

Please cite the release you use, with [CITATION.cff](CITATION.cff); v1.3.0 accompanies the repositioned paper. No journal DOI or software DOI is assigned by this repository. The locally prepared v1.0.0 was not previously published at this GitHub address; v1.1.0 is the first public release here.

The code is publicly accessible. A software reuse license has not yet been selected by the authors; public visibility alone does not grant an unrestricted software license. Third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External datasets retain their source terms and attribution requirements.
