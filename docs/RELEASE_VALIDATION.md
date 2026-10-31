# Release validation — v1.2.0

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
