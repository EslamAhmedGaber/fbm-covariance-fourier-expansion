# Release validation — v1.3.0

Checked on 6 November 2026 in the release working tree with Python 3.12.

| Check | Observed result |
|---|---|
| `python code/tests/test_covfourier.py` | 19/19 passed |
| `python code/tests/test_design.py` | 5/5 passed |
| Original manuscript numerical-claim audit | 167/167 passed |
| New-applications numerical-claim audit (three snippets updated to the revised wording, two checks added) | 117/117 passed |
| Repositioning audit (law of quadratic functionals, scaled core, proof check, Figure 6, Table 4) | 37/37 passed |
| Manuscript citation consistency | 39 listed and 39 cited; first-citation order matched |
| Figures 1-10 and Supplementary Figures S1-S2 | regenerated in this tree from the stored results; plotted data identical to the manuscript build |
| `scaled_conditioning.py`, `law_summary.py` rerun in this tree | summary and tables identical; scaled-core condition numbers agree to 5e-15 |

New computations of this version: `L0_prop1_proof_check.txt`, `L1_spectra.npz`, `L1_spectra_meta.json`, `L2_brackets*.json`,
`L2_cdf_band.json`, `L3_exact_bm.json`, `L4_discretized.json`, `L5_monte_carlo.json`, `L6_scaled_conditioning.json`,
`L7_inversion_check.json`, `L8_summary.json`, `L9_timing.json` and `law_tables/`. The heavy law computations (spectra,
discretized laws, Monte Carlo, de Hoog inversion) ran in the development workspace and were not rerun in this tree.
The numerical-claim audits check agreement of quoted values with the recorded data, not independent proof of the mathematics.
`MANIFEST_SHA256.json` was regenerated from the normalized Git blobs.

---

# Release validation — v1.2.0 (previous release)

Checked again on 31 October 2026 in an isolated release staging directory using Python 3.12.4 and the dependencies in `requirements-tested.txt`. Both test suites and both private-manuscript numerical-claim audits passed; 37 citations were checked, and the abstract contains 248 words. The v1.1.0 tag and its validation record are unchanged.

| Check | Observed result |
|---|---|
| `python code/tests/test_covfourier.py` | 19/19 passed |
| `python code/tests/test_design.py` | 5/5 passed |
| Original manuscript numerical-claim audit | 167/167 passed |
| New-applications numerical-claim audit (now including the worked example, Table 5, the design under `H = 1/2` and Supplementary S3) | 115/115 passed |
| Manuscript citation consistency | 37 listed and 37 cited; no missing or uncited keys; first-citation order matched |
| Figures 1-10 | regenerated from the stored results with the scripts in `figures/src/` |

New computations of this version: `P1_worked_example.json`, `A7_model_selection.json`, `A7_borehole_regional_H075.json`, `A6_borehole_design_H05.json` and the re-run `A7_relog_forecast.json`. The numerical-claim audits ran against the private manuscript and supplement; they check agreement of quoted values with the recorded data, not independent proof of the mathematics or external validity of the climate model.

The final-revision author regenerated the figures and ran the new experiments. Publication staging reused those outputs; it did not rerun every heavy experiment or render all figures. Manuscript audits used a private copy of the main text and supplement plus the ignored coordinate cache. These private files and caches are excluded from the release archive. `MANIFEST_SHA256.json` was regenerated from the normalized Git blobs.
