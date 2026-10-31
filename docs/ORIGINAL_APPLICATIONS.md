# Original applications and figure commands

## Reproduction commands

Run from the repository root. Every script writes into `results/` (or `figures/`). Times are the wall-clock
times recorded for the stored runs (or, where marked, for a regeneration run) on the authors' 12-core Windows
workstation (16 GiB); "not recorded" means that the run did not log its total time. Timings differ between runs (see `REPRODUCTION_ORIGINAL.md`).

| Paper item | Command | Writes | Recorded time |
|---|---|---|---|
| Regression tests (19) | `python code/tests/test_covfourier.py` | console | about 11 s |
| Sec. 4, Gram bridge | `python code/studies/gram_bridge_check.py` | `A1_gram_vs_spectral.json` | 239 s |
| Suppl. S1, Table S1 (whole line) | `python code/studies/w1_whole_line.py` | `W1_whole_line.json` | 20 s |
| Sec. 5, Fig. 2 (two-sided bounds) | `python code/studies/two_sided_checks.py` | `A3_weighted_inequality.json`, `A4_two_sided_bound_checks.json`, `A5_two_sided_rates_inputs.json` | not recorded |
| Sec. 5-6, Fig. 3a (raw Gram conditioning) | `python code/studies/b1_conditioning.py` | `B1_augmented_conditioning.json` | not recorded |
| Sec. 6, 90-digit references | `python code/reference_90digit/run_scaled_projection.py`, then `python code/reference_90digit/extract_reference.py` | `scaled_projection_257.json`, `reference_highprecision_D257.json` | 359 s (first step) |
| Sec. 6, high-precision validation | `python code/studies/b2_highprecision_check.py` | `B2_vs_90digit_reference.json` | not recorded |
| Sec. 6, Fig. 3b,c (scaling to D = 32769) | `python code/studies/scaling_study.py` | `S1_scaling_study.json` | 796 s |
| Sec. 6, sampled inputs | `python code/studies/s2_callable_scaling.py`; `s2_auxiliary_length_indicator.py`; `s2_reorthogonalization_ablation.py` | `S2_callable_scaling.json`; `S2_auxiliary_length_indicator.json`; `S2_reorthogonalization_ablation.json` | 254 s; not recorded; not recorded |
| Sec. 6, Table 4 (mixed inputs) | `python code/studies/s4_mixed_interface.py` | `S4_mixed_interface.json` | not recorded |
| Suppl. S2, Table S2 (endpoint singularities) | `python code/studies/s3_endpoint_remedy.py` | `S3_endpoint_remedy.json` | not recorded |
| Sec. 7.2, Fig. 5 | `python code/applications/app4_kl_spectrum.py`; `app4_analysis.py`; `app4_richardson.py` | `A4_kl_spectrum.json`; `A4_analysis.json`, `A4_eigs_J4096.npz`; `A4_richardson.json` | not recorded; 165 s; not recorded |
| Sec. 7.3, Fig. 6 | `python code/applications/app5_weighted_kl.py`; `app5_mc_reference.py`; `app5_mc_discretization_bias.py`; `app5_refinement_check.py` | `A5_weighted_kl.json`; `A5_mc_reference.json`; `A5_mc_discretization_bias.json`; `A5_refinement_N32768.json` | six cases of 46-60 s; 351 s; 23 s; not recorded |
| Sec. 7.4, Figs. 7-8 | `python code/applications/app2_noisy_reconstruction_v2.py`; `app2_gp_baseline_strong.py`; `app2_timing_and_summary.py retime`, `... m12000`, `... summary`; `app2_memory_probe.py M` (M = 16000, 32000, 64000, 128000); `app2_reuse_and_diagnostics.py [diagnostic\|reconstruction\|incremental\|all]` | `A2_noisy_reconstruction_v2.json`; `A2_gp_baseline_strong.json`; `A2_cf_retime_H075.json`, `A2_m12000_H075.json`, `A2_v2_summary.json`; `A2_memory_probe_mM.json`; `A2_residual_diagnostic.json`, `A2_reconstruction.json`, `A2_incremental_reuse.json` | memory probes 48-219 s each; others not recorded |
| Sec. 7.1, Fig. 4 | `python code/applications/app1_memory_responses.py`; `app1_rerun_cf_v2.py`; `app1_endpoint_ablation.py` | `A1_memory_responses.json`; `A1_memory_responses_v2.json`; `A1_endpoint_ablation.json` | regeneration run: timed steps 396 s; 258 s; not recorded |
| Figures 2-8 | `python figures/src/make_figures.py [fig2 ... fig8]` | `figures/fig2`-`fig8` (PDF, SVG, PNG), `figures/data/` | 4-7 s per figure with a warm TeX cache |
| Layout check of Figures 2-8 | `python figures/src/check_layout.py [fig2 ... fig8]` | figures as above; collisions listed on the console | as above |
| Figure 1 (the CovFourier procedure) | `python figures/src/build_fig1.py` | `figures/fig1.*`, `figures/data/fig1_build.json` | not recorded |

The figure scripts only replot stored results. `code/applications/app2_noisy_reconstruction.py` is the first
version of the sensor study, kept for provenance; its output is not used. `figures/src/build_procedure_example.py`
(after `make_fig1_glyphs.py`) builds the former Figure 1, a worked procedure example, into `figures/extra/`.

## Figure typography

`make_figures.py` uses LaTeX typography (`text.usetex`) when `latex` and `dvipng` are on the path; set
`FIGURES_USETEX=0` to use matplotlib's own mathtext instead (same layout, different glyphs). The figures of the
paper were built with TeX Live 2025. With a warm TeX cache each figure takes 4-7 s; a first build compiles every
label once (about 200 s for Figures 2-8 on the authors' workstation). Matplotlib resolves TeX font files through a
persistent `luatex` process; where that process cannot be started (for example in restricted sandboxes),
matplotlib falls back to one `kpsewhich` call per file, and a build can then take many minutes per figure.

## Contents

- `code/covfourier/`: the package. `gram.py` closed-form covariance inner products; `engine.py`, `batch.py`,
  `large.py`, `solvers.py` projection, white coordinates and residual energies; `endpoint.py` graded endpoint rule;
  `kummer.py` confluent hypergeometric values; `wholeline.py` whole-line realization; `spectral_check.py`
  independent spectral route; `moments.py` exact moments of validation inputs; `comparators.py` competing methods.
- `code/tests/`: regression tests (standalone runner; `pytest` is not required).
- `code/studies/`: studies behind Sections 4-6 and Supplementary S1-S2.
- `code/applications/`: the worked example of Section 3 and the applications of Section 7.
- `code/reference_90digit/`: arbitrary-precision reference implementation (see its README).
- `results/`: stored results quoted in the paper and read by the figure scripts.
- `figures/`: Figures 1-8 (PDF, SVG, PNG), plotted data (`figures/data/`), scripts and TeX sources (`figures/src/`),
  and the former Figure 1 (`figures/extra/`).
- `REPRODUCTION_ORIGINAL.md`: reproduction notes; `MANIFEST_SHA256.json`: release hashes.

Manuscript drafts, correspondence, review material, credentials and unrelated projects are excluded.

