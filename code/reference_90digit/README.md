# Arbitrary-precision reference implementation (90 digits)

These modules are the authors' arbitrary-precision implementation (mpmath, 90 significant digits, nested Cholesky
factorization of the raw Gram matrix) of the same projection. They produce the reference residual energies against
which study B2 (`code/studies/b2_highprecision_check.py`) checks the double-precision formulation of Section 5.

| File | Role |
|---|---|
| `run_scaled_projection.py` | driver; writes `results/scaled_projection_257.json` (about 6 min) |
| `extract_reference.py` | writes `results/reference_highprecision_D257.json`, the 15 reference values read by study B2 |
| `reusable_expansion.py`, `general_applications.py`, `automatic_expansion.py`, `general_expansion.py`, `fractional_projection.py`, `smooth_enrichment.py`, `closed_gram.py` | library modules |

The library modules are included unchanged. Five of them are listed with SHA-256 hashes under `source_hashes` in
`results/scaled_projection_257.json`, which the original driver wrote; the copies here have the same hashes. The
driver differs from the original only in its output path and in the location used for the hash record. In the
comments of these modules, "Paper 1" denotes the companion manuscript on the fractional Wiener chaos solver
(closed-form cosine Gram matrices) and "Paper 2" this work.

Run from this folder:

    python run_scaled_projection.py --terms 257
    python extract_reference.py
