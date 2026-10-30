# Scope and limitations

## Mathematical and numerical claims

The construction concerns deterministic Wiener integrands for fractional Brownian motion, with `1/2 <= H < 1`, in the stated covariance space. It does not assert automatic certification for every arbitrary function supplied through finitely many samples. Whole-line and finite-interval realizations have different domains and must not be conflated.

Theoretical projection identities, computed residuals, Fourier-stage resolution estimates and floating-point allowances play different roles. Numerical brackets and design gaps must be interpreted with the error allowances used in the associated experiment. The implementation is not interval-arithmetic verified software. A passing stored-number audit checks consistency with result files, not independent validity of a theorem or a statistical model.

The design variance bracket is an orthogonal-projection result; the reusable covariance coordinates make its terms practical to evaluate across many candidate sensors. Greedy selection alone does not prove global optimality. The reported suboptimality bound also uses the relaxation lower bound and the truncation upper bound.

## Performance comparisons

Recorded timings are hardware-, workload- and configuration-dependent. A dense covariance storage calculation is distinct from measured peak process memory. Comparisons to exact GP means were performed in the feasible matched experiments, not with a dense 128,000-sensor GP. Extrapolated dense running times and Monte Carlo requirements are not completed runs. Specialized product quadrature can outperform the expansion on stiff kernels.

## Borehole case study

- The real-data likelihood favors `H = 1/2` among the tested H values. This study demonstrates use of the framework; it does not establish empirical superiority of fractional long memory.
- The first white-noise-only model is an adverse development step. Regional climate, local paths and geological components improve fit but do not establish a correct physical model.
- Target-error estimates acknowledge that ground temperatures, gridded air temperatures and station observations are different quantities. Some choices were developed after inspecting discrepancies; the subsequent comparisons should not be described as an untouched prospective validation set.
- Reported 90% site coverage of about 80% still undercovers. Japan remains discrepant; the urban-station checks support a possible explanation rather than proving the discrepancy fully resolved. Splitting China into regions is exploratory.
- Only five re-logging pairs are available: the reconstruction improves on no-change for three, and the Brownian prior has higher predictive density than the fractional comparison for four. This is limited, mixed forecasting evidence.
- The real-network design run in `A7_network_design.json` has a loose certificate; the strong design-gap result belongs to the controlled synthetic heat experiment.

Adverse outputs are retained for transparency. Filenames containing `partial`, `window500`, or the earlier `v1` identify exploratory or sensitivity runs rather than replacements for the main stored results. Figure-to-result mappings are in `NEW_APPLICATIONS.md`.
