# Covariance-orthonormal Fourier expansion for fractional Brownian motion

Research software accompanying *A reusable covariance-orthonormal Fourier expansion for fractional Brownian motion: two-sided bounds, stable computation and measurement design*, by Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman and Sroor M. Elnady.

**v1.1.0 — 30 October 2026.** This release includes the two new measurement-design and borehole applications. It is a research-software release, not a claim of journal acceptance.

## One representation, several tasks

For fractional Brownian motion with `1/2 <= H < 1`, the method orthonormalizes a fixed nested family of cosines and two polynomial boundary modes in the covariance inner product. Each admissible deterministic integrand is mapped to shared white coordinates by the same coefficient rule. The coordinates support Wiener-integral simulation, covariance calculations, Gaussian conditioning and measurement design; residual energies quantify the projection error.

The mathematical construction and its finite-precision implementation are distinguished throughout: arbitrary black-box samples cannot by themselves certify an unresolved function. Reported numerical certificates depend on resolved Fourier coefficients and the stated numerical-error allowances. At `H = 1/2`, the covariance norm becomes the ordinary `L2` norm; omitting the enrichment recovers the classical cosine system.

## Results included

| Task | Evidence in this release |
|---|---|
| Stable expansion | Double-precision, matrix-free studies up to `D = 32,769`; comparisons with independent references |
| Spectra and quadratic functionals | KL eigenvalue studies, weighted functionals including `w(t) = 1/t`, deterministic truncation brackets and Monte Carlo comparisons |
| Noisy sensors | Experiments up to 128,000 sensors, measured peak RAM 0.68 GiB; a dense float64 covariance alone would occupy about 122 GiB |
| Multiple responses | Smooth and stiff kernels, including cases where specialized quadrature is preferable |
| Measurement design | A posterior-variance bracket and a computable bound on design suboptimality; 21,497-candidate synthetic heat experiment, with reported relative gaps 0.021–0.039% under the stated model and numerical allowances |
| Real borehole data | 690 selected profiles; regional reconstruction, external temperature comparisons and adverse results retained |

These numbers describe particular stored experiments, not universal speedups or accuracy guarantees. The real-data likelihood favors `H = 1/2` among the tested values. After target-error adjustment, nominal 90% site intervals cover about 80%; Japan remains discrepant. The real-network design experiment does not produce a useful tight certificate. See [limitations](docs/LIMITATIONS.md).

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
- [New design and real-data applications, Figures 9–10](docs/NEW_APPLICATIONS.md).
- [External data download and attribution](04_DATA/README.md).
- [Checks performed for this release](docs/RELEASE_VALIDATION.md).

Stored result files are included. Large regenerable caches and external raw datasets are excluded. Download the exact external snapshots, checking their SHA-256 hashes, with:

```sh
python tools/fetch_data.py --dataset all
```

For Figures 2–10 from stored results:

```sh
python figures/src/make_figures.py
python figures/src/make_figures_design.py fig9
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

Please cite **v1.1.0**, using [CITATION.cff](CITATION.cff). No journal DOI or software DOI is assigned by this repository. The locally prepared v1.0.0 was not previously published at this GitHub address; v1.1.0 is the first public release here.

The code is publicly accessible. A software reuse license has not yet been selected by the authors; public visibility alone does not grant an unrestricted software license. Third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External datasets retain their source terms and attribution requirements.
