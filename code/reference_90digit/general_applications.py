"""General shared-noise outputs and Gaussian conditioning from fixed features.

No integrand-name branches. Numerical quadrature diagnostics are estimates.
Covariance envelopes are conditional on exact projection features and valid
residual norm upper bounds; they are not automatically certified by this code.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import cho_factor, cho_solve, solve_triangular

def _symmetric(a):
    return (a + a.T) / 2

def _finite_matrix(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(name + " must be a finite matrix")
    return a

@dataclass
class OutputExpansion:
    engine: object
    coefficients: np.ndarray
    diagnostics: dict

    @property
    def covariance(self):
        return self.coefficients @ self.coefficients.T

    def truncate(self, terms):
        if not isinstance(terms, int) or not 1 <= terms <= self.coefficients.shape[1]:
            raise ValueError("Unavailable prefix")
        return OutputExpansion(self.engine, self.coefficients[:, :terms].copy(),
                               {**self.diagnostics, "prefix_of": self.coefficients.shape[1]})

    def sample(self, count, rng):
        if not isinstance(count, int) or count < 1:
            raise ValueError("Positive sample count required")
        return rng.standard_normal((count, self.coefficients.shape[1])) @ self.coefficients.T

    def raw_coefficients(self):
        return np.array([self.engine.raw_coefficients([str(x) for x in row])
                         for row in self.coefficients])

def expand_outputs(engine, functions, tolerance=1e-8, max_intervals=4096):
    """Same coefficient integrals, sharing each dual-weight evaluation."""
    functions = tuple(functions)
    if not functions or tolerance <= 0:
        raise ValueError("Nonempty callable sequence and positive tolerance required")
    calls = 0
    def integrand(t):
        nonlocal calls
        calls += 1
        values = np.array([float(f(t)) for f in functions])
        if not np.all(np.isfinite(values)):
            raise ArithmeticError("Nonfinite input at an interior point")
        return np.outer(values, engine.dual_values(t))
    T = float(engine.T)
    points = np.linspace(0, T, max(8, math.ceil((engine.D-3)/4))+1)[1:-1]
    start = time.perf_counter()
    runs = []
    for factor in [1., .2]:
        c, err, info = quad_vec(integrand, 0, T, epsabs=tolerance*factor,
            epsrel=tolerance*factor, norm="2", points=points,
            limit=max_intervals, full_output=True)
        runs.append((c, err, info))
    c, err, info = runs[-1]
    changes = np.linalg.norm(c-runs[0][0], axis=1)
    # Explicit row diagnostics: a well-resolved large output does not by itself
    # mark a smaller output resolved.
    scales = np.maximum(1., np.linalg.norm(c, axis=1))
    row_resolved = (changes <= tolerance*scales) & (err <= tolerance*scales)
    success = all(r[2].success for r in runs) and np.all(row_resolved)
    return OutputExpansion(engine, c, {
        "status": "coefficients_numerically_resolved" if success else "coefficient_quadrature_unresolved",
        "outputs": len(functions), "terms": engine.D,
        "seconds": time.perf_counter()-start, "tolerance": tolerance,
        "shared_dual_evaluations": calls, "scalar_input_evaluations": calls*len(functions),
        "quadrature_error_estimate": float(err),
        "row_refinement_differences": changes.tolist(),
        "row_resolved": row_resolved.tolist(),
        "quadrature_status": [int(r[2].status) for r in runs],
        "qualification": "Estimated coefficient accuracy, not a certified truncation error."
    })

def condition_covariance(covariance, targets, noise_covariance, observations):
    """Targets precede observations in one joint latent covariance matrix."""
    K = _finite_matrix(covariance, "covariance")
    if K.shape[0] != K.shape[1] or not 1 <= targets < len(K):
        raise ValueError("Square covariance and a nonempty target/observation split required")
    if not np.allclose(K, K.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Covariance must be symmetric")
    n = len(K)-targets
    noise = _finite_matrix(noise_covariance, "noise covariance")
    y = np.asarray(observations, dtype=float)
    if noise.shape != (n,n) or y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError("Incompatible observations or noise")
    if not np.allclose(noise, noise.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Noise covariance must be symmetric")
    # Positive noise is the current contract; do not silently add a ridge.
    cho_factor(noise, lower=True)
    S = _symmetric(K[targets:, targets:]+noise)
    factor = cho_factor(S, lower=True)
    A = cho_solve(factor, K[targets:, :targets]).T
    V = _symmetric(K[:targets, :targets]-A @ K[targets:, :targets])
    return {"mean": A@y, "covariance": V, "gain": A,
            "observation_covariance": S, "observation_condition_number": float(np.linalg.cond(S))}

def residual_conditioning_envelope(coefficients, targets, noise_covariance, observations, residual_norm_bounds):
    """Conditional theorem interface; bounds supplied by caller, never inferred.

    If C is the exact projection feature matrix and eps_j bounds ||(I-P)f_j||,
    the true conditional covariance lies between lower and upper. Estimated
    coefficients/eps require a separate numerical error analysis.
    """
    C = _finite_matrix(coefficients, "coefficients")
    eps = np.asarray(residual_norm_bounds, dtype=float)
    if eps.shape != (len(C),) or np.any(eps < 0) or not np.all(np.isfinite(eps)):
        raise ValueError("One finite nonnegative residual norm bound per input required")
    K0 = C@C.T
    lower = condition_covariance(K0, targets, noise_covariance, observations)
    # Weighted Cauchy-Schwarz: R <= sum(eps) diag(eps).
    majorizer = np.diag(np.sum(eps)*eps)
    upper = condition_covariance(K0+majorizer, targets, noise_covariance, observations)
    A = lower["gain"]
    d = eps[:targets]+np.abs(A)@eps[targets:]
    nu = np.linalg.eigvalsh(np.asarray(noise_covariance))[0]
    basic_mean_bounds = d*np.linalg.norm(eps[targets:])*np.linalg.norm(observations)/nu
    # Observation whitening bounds lambda/(1+lambda)^2 for residual/noise
    # eigenvalues. This retains the residual size without an unnecessary 1/nu
    # penalty when the observed vector lies in a well-determined direction.
    L = np.linalg.cholesky(lower["observation_covariance"])
    eg = eps[targets:]
    Pg = np.diag(np.sum(eg)*eg)
    left = solve_triangular(L, Pg, lower=True)
    whitened = solve_triangular(L, left.T, lower=True).T
    rho = max(0., float(np.linalg.eigvalsh(_symmetric(whitened))[-1]))
    h = rho/(1+rho)**2 if rho <= 1 else .25
    z = solve_triangular(L, np.asarray(observations), lower=True)
    refined_mean_bounds = d*np.sqrt(h*np.dot(z,z))
    mean_bounds = np.minimum(basic_mean_bounds, refined_mean_bounds)
    return {"mean": lower["mean"], "lower_covariance": lower["covariance"],
            "upper_covariance": upper["covariance"], "mean_error_bounds": mean_bounds,
            "basic_mean_error_bounds": basic_mean_bounds, "whitened_residual_radius": rho,
            "residual_majorizer": majorizer,
            "qualification": "Conditional on exact projection coefficients and valid residual norm upper bounds."}

