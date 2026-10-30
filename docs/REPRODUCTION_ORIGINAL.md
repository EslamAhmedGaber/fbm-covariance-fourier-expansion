# Historical reproduction record: original four applications

These notes describe the original assistant's runs, not the environment of every result in v1.1.0. See NEW_APPLICATIONS.md and RELEASE_VALIDATION.md for this release.

# Reproduction notes

## What reproduces exactly and what does not

- **Accuracy fields reproduce; timing fields do not.** Regenerating the A1 studies (Sec. 6.4) reproduces all 1083
  (first set) and 1803 (second set) per-configuration accuracy fields exactly; regenerating the A2 sensor study
  (Sec. 6.3) reproduces all 1394 accuracy fields exactly. Wall-clock times differ between runs (for A1 the median
  ratio of new to stored times was 0.70-1.43 per configuration group), and selections that depend on timings
  (the "matched" picks at equal accuracy) can change with them. The paper states timing variability of up to
  about 40%. The stored result files are the ones quoted in the paper.
- **Random draws** use fixed seeds (`numpy.random.default_rng`) set in each script. On the tested software stack
  sampled quantities reproduce; other BLAS builds can change the last digits.
- **Platform.** All stored results were produced on Windows 10 with Python 3.11.16 (`requirements-tested.txt`),
  12 cores and 16 GiB of memory.

## Settings that the stored results depend on

- **A1 endpoint-correction rule.** The stored A1 files were produced with the first, ratio-only rule of the
  endpoint-corrected Fourier stage: the correction is skipped when its polynomial part exceeds 100 times the
  maximum of the input, without the tail-energy test. The A1 scripts therefore construct
  `BatchProjector(..., rule='ratio', max_ratio=100)`. The default rule of the package (the a posteriori switch
  described in Sec. 5 of the paper) is used everywhere else; `code/tests/test_covfourier.py` tests both rules.
- **Excluded one-time cost.** Each Hurst index needs 30-digit Chebyshev tables of the confluent hypergeometric
  factor (`warm_tables(H)`, about 1 s, shared by all n, N and inputs). The application scripts build them before
  timing, and the comparisons of Secs. 6.3-6.4 exclude them, as stated in the paper.
- **A2 extrapolated GP times.** The extrapolation of exact-GP times beyond the largest feasible m uses a
  non-negative least-squares fit; its coefficients equal the stored ones to 6e-16.
- **Large projections.** By default (`reorth=True`) `LargeProjector` re-orthogonalizes the two enrichment
  directions against the cosine space with one extra solve; the ablation
  `code/studies/s2_reorthogonalization_ablation.py` shows the orthogonality defect of a single solve.

## Reference values

The 90-digit reference values (Sec. 5) come from the independent arbitrary-precision implementation in
`code/reference_90digit/` (mpmath): `run_scaled_projection.py` writes `results/scaled_projection_257.json`, and
`extract_reference.py` writes `results/reference_highprecision_D257.json`, which `b2_highprecision_check.py`
compares with the double-precision engine. The Monte Carlo references of Sec. 6.2 use exact Davies-Harte paths;
`app5_mc_discretization_bias.py` computes the exact time-discretization bias of those estimators.

## Figures

The figure scripts read only `results/`. `figures/src/check_layout.py` rebuilds Figures 2-8 and reports
collisions between text and data and between text boxes; the release figures pass it. See the README for the
TeX setup and for `FIGURES_USETEX=0`.
