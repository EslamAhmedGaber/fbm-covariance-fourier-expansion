# Law of quadratic functionals (v1.3.0)

Section 6.4 of the paper turns the bracketed spectra of Sections 6.2-6.3 into the probability law of
`Q_w = int_0^1 w(t) B_H(t)^2 dt` with two-sided bounds (Proposition 4): with lower eigenvalues `b_n <= lambda_n` and the
exact trace, `S = sum b_n xi_n^2 <= Q` and the discarded mass `R = Q - S >= 0` has mean `Delta`, so
`F_S(x - eps) - exp(-(sqrt(eps/Delta) - 1)^2 / 2) <= F_Q(x) <= F_S(x)`.

| Script (`code/studies/`) | Result (`results/`) | Content | Run time (development machine) |
|---|---|---|---|
| `law_study.py spectra` | `L1_spectra.npz`, `L1_spectra_meta.json` | eigenvalues for `w = 1` (stored Galerkin, J = 4096) and `w = 1/t` (CovFourier coordinates, J = 1024, N = 16384 and 32768) | about 25 min |
| `law_study.py disc` | `L4_discretized.json` | exact law of the discretized functional on 1024-8192 grid points (reference independent of CovFourier) | about 12 min |
| `law_study.py mc` | `L5_monte_carlo.json` | Monte Carlo, 10^6 exact Davies-Harte paths per case (`MC_CASES=inv_t:0.55 MC_OUT=...` runs one case) | 6 min (w = 1) to 24 min (w = 1/t) per case |
| `run_bracket_one.py`, `run_bracket_inv.py` | `L2_brackets*.json`, `L2_cdf_band.json`, `L3_exact_bm.json` | quantile and distribution-function brackets, moment-matched approximations, test sizes, exact law at `H = 1/2` | a few minutes |
| `check_inversion_dehoog.py` | `L7_inversion_check.json` | de Hoog inversion in 30-digit arithmetic at every bracket end | about 13 min |
| `law_summary.py` | `L8_summary.json`, `law_tables/` | summary, references, Monte Carlo cost to match a bracket; Table 4 and Supplementary Table S4 (bracket ends rounded outward) | seconds |
| `scaled_conditioning.py` | `L6_scaled_conditioning.json` | condition number of the scaled core for Figure 3a | about 1 min |
| `check_prop1_proof.py` | `L0_prop1_proof_check.txt` | 30-digit quadrature check of the closed cosine Gram matrix (Proposition 1) at H = 0.7, T = 1.7 | about 1 min |

`L9_timing.json` records the measured cost of the brackets. Figure 6 and Supplementary Figures S1-S2 are drawn by
`figures/src/make_figures.py fig6 figS1 figS2`.

The brackets are computed brackets: eigensolver rounding is covered by an allowance subtracted from every eigenvalue,
and the Laplace inversion is checked by a second route rather than bounded. The width of the upper side follows
`Delta`; for `w = 1/t` at `H = 0.55` the brackets are 0.2-1.7% wide, and at the median 8e4 Monte Carlo paths give an
interval as narrow.
