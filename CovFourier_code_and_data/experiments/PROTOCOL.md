# Experiment protocol — CovFourier

> Earlier analyses are named here by their software release (CovFourier package v1.3.0) or as the earlier
> analysis of the same data; designs, criteria and dates are as recorded.

Written 12 November 2026, before any experiment was run (only the unit tests of
`code/cfgp/core.py` listed in Section 0.6 had been executed). Results that miss a criterion are reported,
not tuned away; failed attempts stay in `results/` and in `logs/`. Changes made after a run are appended
in Section 3 with date and reason.

## 0. Common conventions

### 0.1 Model
Observations y_i = σ I_H(F_i) + h_iᵀβ + τ ε_i, i = 1, …, m, ε_i i.i.d. N(0,1), on [0, T] with T = 1,
1/2 ≤ H < 1. K_il = ⟨F_i, F_l⟩_H is the exact covariance of the Wiener integrals. Nuisance h_i = (1, x_i − 1/2)
with x_i the location of observation i in [0, 1] (point, window centre, rescaled depth).
* Gaussian prior β ~ N(0, Σ_β), Σ_β = I₂: evidence of the model with covariance K + hΣ_βhᵀ + τ²I. The ordering
  K = ΓΓᵀ + R applies unchanged with Q = ΓΓᵀ + hΣ_βhᵀ, because R is untouched.
* Flat prior (restricted likelihood, REML): log N(Aᵀy; 0, Aᵀ(K + τ²I)A) for an orthonormal basis A of h^⊥,
  written without A through log det(AᵀCA) = log det C + log det(hᵀC⁻¹h) − log det(hᵀh). Congruence keeps
  AᵀRA ⪰ 0 and tr(AᵀRA) ≤ tr R, so every bound below carries over with v = P_C y.

### 0.2 Observation kinds (designs fixed across replicates; data vary through y only)
* P — point values B_H(t_i): t_i i.i.d. U(0.01, 1) (seed 11).
* W — box-window averages w_i⁻¹∫_{a_i}^{b_i} B_H: centres U(0.03, 0.97), widths U(0.005, 0.06), clipped to [0, 1] (seed 12).
* S — smooth kernels: subsurface temperatures of a half-space driven by a surface-temperature history
  B_H on the window 1500–2030 (rescaled to [0, 1]), diffusivity 31.5576 m² yr⁻¹; two loggings (1985 and 2015)
  of m/2 equally spaced depths in 10–500 m. Integrands are represented exactly by their endpoint-corrected
  8192-term cosine series plus a quadratic (the "N-model", covfourier.batch); all covariances of kind S are
  exact for the N-model. The N-model is checked against the independent quadrature `covfourier.khref.KHQuad`
  on a subset (tolerance 1e-9 absolute).

### 0.3 Features and bounds
* CF: n cosines whitened by the Cholesky factor of the closed-form cosine Gram (cited from CosChaos), plus the
  two H-orthonormal residual directions of t and t² (Naux = 4n cosine terms, re-orthogonalised twice);
  D = n + 2. Coordinates from exact moments ⟨F_i, h_k⟩_H (closed forms for P and W; N-model for S);
  residual energies e_i² = ‖F_i‖²_H − |γ_i|² (negative values clipped to 0 and counted).
* Bounds, all evaluated with the same Γ, e and τ:
  - L — collapsed lower bound log N(y; 0, C) − tr R/(2τ²), C = Q + τ²I (Titsias; cited, not new).
  - U0 — trace upper bound: −½ log det C − ½ yᵀ(C + tr R·I)⁻¹y − (m/2) log 2π (cited form).
  - U1 — per-vector diagonal bound: quadratic term yᵀC⁻¹y − (|v|ᵀe)², v = C⁻¹y (|R_il| ≤ e_i e_l).
  - U2 — per-vector majorant: quadratic term yᵀC⁻¹y − β(v), β(v) ≥ vᵀRv from the cosine representation of
    the residual combination Σv_i r_i: exact Gram on Naux cosines plus the L² tail through Λ_N of the weighted
    cosine inequality (cited); exact (no tail) for kind S.
  - U = min(U0, U1, U2). W — working model with covariance Q + diag(e²) + τ²I (not a bound).
* Baselines: EXACT (dense Cholesky); IP-VFE (inducing values B_H(z_j), M = D, uniform grid z_j = j/M,
  smallest jitter in {0, 1e-12, …, 1e-4}·mean diag K_uu that factorises; jitter and cond(K_uu) recorded);
  IP-FITC (its diagonal-corrected model); IP-opt (pseudo-inputs optimised, M ≤ 32); COS (enrich = False,
  D cosines, H-orthogonal projection); COS-classical (classical cosine truncation; not a valid bound);
  ORACLE (top-D eigenpairs of K).

### 0.4 Margin, seeds, units, fairness
* A violation is L − ℓ > δ or ℓ − U > δ with δ = 1e-6 + 1e-9·|ℓ| nats (ℓ the exact log evidence). E11 checks
  that the floating-point error of every term is below δ.
* Data seeds: 20261112 + 1000·kind + 100·iH + 10·inoise (+ offsets stated per experiment). Designs: seeds 11, 12.
* Evidence gaps in nats and nats per observation; ρ = tr R/τ² reported next to every gap.
* Same data sets for all methods; same rank D = M; double precision; timings with one BLAS thread when they
  are compared; preparation time reported separately.

### 0.5 Noise levels
τ = f · sd_rms with sd_rms = (mean_i K_ii)^{1/2} of the kind and Hurst index, f ∈ {0.1, 0.03, 0.01}.

### 0.6 Unit tests run before this protocol (m = 300, H = 0.75)
Closed-form point and window moments, covariances, cross-covariances, classical coefficients and L² forms
agree with independent quadrature to 1e-11–1e-13; orthonormality defect after re-orthogonalisation ≤ 9e-16
for n ≤ 1024 (8.9e-16; the engine's single pass gave 5e-11 at n = 1024); smallest eigenvalue of R positive;
U2 within a factor 1.2–1.9 of the exact per-vector correction; no bound violated.

## 1. Experiments

### E1 `bracket_exact`
* Question: does L ≤ ℓ ≤ U always hold, and how wide is the bracket?
* Settings: m = 2000; kinds P, W, S; H ∈ {0.55, 0.75, 0.95}; f ∈ {0.1, 0.03, 0.01};
  n ∈ {8, 16, 32, 64, 128, 256, 512, 1024} (D = n + 2); nuisance ∈ {none, Gaussian, REML}; 100 data sets per
  (kind, H, f, nuisance), shared across n.
* Metrics: violations; ℓ − L; U − ℓ for U0, U1, U2, U; |W − ℓ|; ρ; times.
* Success: zero violations in every cell; median(ℓ − L) ≤ 1 nat at the recommended D_rec := the smallest
  grid D with ρ ≤ 1 (computable without ℓ).

### E2 `upper_bound`
* Question: is there an upper bound as close to ℓ as the lower bound?
* Settings: the E1 grid; bounds U0, U1, U2.
* Success: in every E1 cell with ρ ≤ 10, median(U − ℓ) ≤ 3·median(ℓ − L) + 0.01 nats. Otherwise the paper
  states L as the working quantity and U as a safeguard.

### E3 `feature_count`
* Question: do the decay orders predict the D needed?
* Settings: kinds P, W, S; m ∈ {10³, 10⁴, 10⁵} (first m observations of a 10⁵-observation master design drawn
  as in 0.2; S: m/2 depths per logging); H ∈ {0.55, 0.75, 0.95}; f = 0.03; ε ∈ {1, 1e-2, 1e-4}.
* Measured: n_meas(ε) = smallest n with ρ(n) ≤ ε on the grid n ∈ {8·2^{j/2}} ∩ [8, 1024], log–log interpolation.
* Predicted (no projection): ρ_pred(n) = Σ_i 𝒯_n(F̃_i)/(√2 τ²), 𝒯_n(f) = Σ_{k≥n} w_k a_k(f)² (classical
  coefficients; F̃ = F for P and W, the endpoint-corrected integrand for S); the factor 1/√2 is the geometric
  middle of the CosChaos two-sided range [½, 1]. Closed-form asymptotics are also reported: points
  ρ ≈ m c*_H n^{−2H}/(2√2 H π^{1+2H} τ²); windows ρ ≈ (Σ_i w_i⁻²) c*_H n^{−2H−2}/(√2 (1+H) π^{3+2H} τ²).
* Success: n_meas/n_pred ∈ [½, 2] in ≥ 80 % of the cells where both are ≤ 1024; cells beyond 1024 reported.

### E4 `benchmark_rank`
* Question: at equal rank, how do CF, IP-VFE, IP-FITC, IP-opt, COS and ORACLE compare?
* Settings: m = 2000 (E1 designs) plus 500 held-out observations of the same kind drawn jointly; H ∈ {0.55,
  0.75, 0.95}; f = 0.03; rank r ∈ {8, 16, 32, 64, 128, 256, 512} (CF: n = r − 2); 100 data sets.
  m = 20,000: CF and IP-VFE only, data simulated by the Davies–Harte method on a 2^20 grid (points on grid
  nodes; windows and kernels by quadrature of the grid path), 10 data sets, held-out 2000.
* Metrics: ℓ − L (= KL to the exact posterior of the collapsed approximation), U − ℓ, tr R and tr R/floor,
  held-out RMSE and NLPD at the true hyperparameters, cond(K_uu), jitter, time (preparation, features, bound).
* Success: CF has the smaller ℓ − L at equal rank on at least two of three Hurst indices for W and S.
  For P inducing points are expected to be competitive or better at small rank; reported as found.
  A regime table (kind × H × rank: who wins) is produced.

### E5 `learn_hurst`
* Question: does learning with the bound recover the exact estimate, and does the bracket enclose it?
* Settings: kind W, m = 2000; true H ∈ {0.6, 0.75, 0.9}, σ = 1, f = 0.03; 100 data sets each; rank ∈ {16, 64, 256};
  learners EXACT, CF-L, IP-VFE (and CF-W). Grid H ∈ {0.50, 0.51, …, 0.99}; σ² profiled in closed form; κ = τ²/σ²
  by bounded Brent search in log κ ∈ [log 1e-8, log 10]; Ĥ by parabolic refinement at the grid maximum.
  Enclosure ℰ = {grid H : U*(H) ≥ max_H′ L*(H′)}.
* Metrics: Ĥ − Ĥ_exact, σ̂/σ̂_exact, τ̂/τ̂_exact; enclosure width and coverage of the exact grid maximiser; time.
* Success: CF at the recommended rank (smallest rank with ρ ≤ 1 at the true H) has |Ĥ − Ĥ_exact| ≤ 0.005 in
  ≥ 95 % of data sets; coverage 100 %.

### E6 `scale`
* Question: time and memory against m; bracket at scale.
* Settings: first reproduce the stored timing of release v1.3.0 (128,000 Gaussian-window sensors, n = 256: 191 s, 0.68 GiB; tolerance
  ±40 % in time, ±10 % in memory). Then kind W with exact moments, streamed in chunks, m ∈ {1.6e4, 6.4e4,
  1.28e5, 5.12e5, 1e6}, D ∈ {66, 130, 258}; f = 0.03; data by Davies–Harte on a 2^21 grid.
* Metrics: time (preparation, moments, projection, bounds), peak memory, ρ, U − L.
* Success: fitted slopes ≈ 1 in m (0.9–1.1) and ≈ 2 in D (1.6–2.4, preparation excluded); the stored v1.3.0 timing reproduced.

### E7 `real_rivers`
* Question: which H do the CAMELS data prefer, and is the preference larger than the truncation error?
* Settings: reproduce the earlier analysis of these data from its stored script first (+899.37 train, +700.43 test nats vs H = ½).
  Then the bracketed pipeline: exact window moments, κ optimised on [1e-8, 10], σ² profiled, D ∈ {130, 258, 514},
  H grid 0.50:0.05:0.95 (+0.01 steps near the maximum); pooled and per basin; held-out blocks by year.
  Basins: the stored pilot (lowest gauge id per HUC-02 region, first 12 regions; rule fixed before any fit),
  minus basins excluded by the data-completeness rule of that analysis (stated).
* Controls: block pinning (per-block demeaning as in that analysis vs training-mean centring); mean model (constant vs
  constant + trend); noise model (estimated white noise vs fixed small noise); block length (365, 730, 1460 days);
  enclosure of the exact estimate per basin.
* Success: L*(Ĥ) > U*(½) pooled and in a stated fraction of basins; otherwise "the data do not decide".

### E8 `real_boreholes`
* Question: the same for the NOAA borehole profiles (the v1.3.0 analysis found H = ½).
* Settings: the REML pipeline of release v1.3.0 (345 single-history profiles) with the bracket added.
* Success: brackets consistent with H = ½ and excluding strong memory, or a statement of what they exclude.

### E9 `active_selection` — optional (only if pages allow).

### E10 `stress`
* Small noise (f = 0.001), H ∈ {0.51, 0.99}, a singular integrand (t^{−0.3} weight on [0, 0.5]) and interval
  increments, misspecified data (Riemann–Liouville process; H switching from 0.6 to 0.9 at t = ½).
* Report validity (must hold: the bracket concerns the model evidence), widths, and learning under misspecification.

### E11 `precision`
* Kummer function and moments in double precision against 50-digit mpmath on random samples; orthonormality
  defect for n ≤ 1024; all bracket terms for an m = 300 subset of each kind in 50-digit arithmetic given the
  double-precision inputs, n ∈ {8, 64, 512}, three H. Gives the margin used in E1.

### E12 `cost_model`
* Per candidate H: CF preparation, moments, projection, bound; IP-VFE K_uu, K_fu, factorisation, bound.
  Total for a 20-value grid and for a Brent search; what is cached across H.

## 2. Order
E1 → E2 → E3 → E4 pilot (H = 0.75) → E5 → E4 full → E6 → E7 → E8 → E10–E12 → E9 if pages allow.
If E1 or E2 fails: stop, find the bug or adopt the fallback of 04_INSIGHTS_AND_REVIEW.md Part C.

## 3. Changes after runs
(appended with date and reason)

* 2026-11-12, all experiments — lower bound. The diagonal (Hadamard) form Ld = log N(y; 0, C) − ½ Σ log(1 + e_i²/τ²)
  (the collapsed bound of Titsias, ICML 2025) is reported as L; the trace form of 0.3 is stored as gap_L.
  Reason: it is never smaller than the trace form and costs the same.
* E3 — windows and heat kernels with m ∈ {10³, 10⁴} only (time budget); points with m ∈ {10³, 10⁴, 10⁵}.
* E4 — IP-opt: L-BFGS-B on the trace-form bound, ≤ 200 iterations for M ≤ 32 (points, windows); heat kernels
  M ≤ 16 and ≤ 25 iterations (cost of kernel cross-covariances). COS-classical not run for heat kernels.
  The m = 20 000 arm was not run; the scale question is answered by E6.
* E5 — the CF-W learner was dropped. The enclosure over kappa ∈ [1e-8, 10] is the whole H grid, because
  U* → +∞ as kappa → 0 when D < m (log det of Γ Γᵀ + kappa I has m − D eigenvalues kappa). E5b
  (e5b_enclosure.py) computes the enclosure on the part of the grid with rho = tr R / kappa ≤ 10, fixed by the
  design before the data: 31 log-spaced kappa in [tr R(H)/10, 10] per H, ranks 64 and 256, same data as E5.
  E5c (e5c_enclosure.py) repeats it with 41 kappa values at rank 256 (reported) and tests a cell bound over continuous
  kappa, which is too loose to be useful (the log-determinant changes by about (m - D)/2 * d log kappa per cell).
* E6 — the 128 000-sensor timing of release v1.3.0 was not reproduced; superseded by E6 itself. Timings were taken
  while other experiments ran on the same machine (upper bounds).
* E7 — kappa grid extended to 201 log-spaced values in [1e-9, 10] (the estimate sits at the lower end);
  per-basin estimates, margins and enclosures recomputed on the full grid 0.50:0.05:0.90 ∪ 0.91:0.01:0.99
  (e7b_per_basin.py), since the first per-basin pass used the coarse grid only.
* E8 — H values ≥ 0.85 run in separate processes (E8_HS / E8_OUT environment variables), same code path.
* E9 — not run (optional).
* E10 — the singular integrand (t^−0.3 weight) was not run; learning under misspecification with 20 data sets.
* E11 — the 50-digit bracket terms use subsets of 120 observations (50-digit Cholesky cost).
* E12 — timings measured on a lightly loaded machine (one other job running) with one BLAS thread
  (results/E12_cost_model.json); a first measurement under concurrent load is not reported.
* 2026-11-13 (after a review of the full draft) — E7: the analysis with climatology, centring and scaling
  from the training years (protocol control `centre_train`) is reported first, because within-block centring pins each
  cumulative anomaly to zero at the block end; its per-basin estimates on the full grid come from
  `e7c_per_basin_centre_train.py` (results/E7c_per_basin_centre_train.json). No new tuning: the configuration and grid
  were already in E7.
* 2026-11-13 — the sharper quadratic term a²/(a+b) (Remark 1 of the paper) was checked on the stored E1 window
  evaluations at H = 0.75 (widths at D = 10, 34, 130, 258); the reported experiments keep the bound of Theorem 2.
* 2026-11-13 — Figure 3 shows the trace bound U0 computed from the same features, and all 648 settings in panel (d).
