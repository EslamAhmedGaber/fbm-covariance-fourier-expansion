"""Additional exact moments and polynomial enrichment for Paper 2.

Endpoint subtraction is established approximation methodology (Huybrechs,
Iserles and Norsett, 2011); no priority claim is made for that idea.
All integrand products here are bilinear, including complex exponential ones.
"""
import mpmath as mp
import fractional_projection as fp


def exp_cross(H, a, b, T=1):
    """Bilinear <exp(a*t),exp(b*t)>_H, with complex parameters allowed."""
    H, T = fp.parameters(H, T)
    a, b = mp.mpc(a), mp.mpc(b)
    z = a+b
    if H == mp.mpf('0.5'):
        return T if z == 0 else mp.expm1(z*T)/z
    beta = 2*H-1
    if z == 0:
        return H*beta*T**(beta+1)*mp.beta(beta, 2)*(
            mp.hyp1f1(beta, beta+2, a*T)+mp.hyp1f1(beta, beta+2, -a*T))
    def F(w):
        return T**beta/beta*mp.hyp1f1(beta, beta+1, w*T)
    return H*beta/z*(mp.exp(z*T)*(F(-a)+F(-b))-F(a)-F(b))


def exp_power(H, p, lam=1, T=1):
    """Exact-identity evaluation by differentiating the entire moment in b."""
    if not isinstance(p, int) or p < 0:
        raise ValueError('p must be a nonnegative integer')
    return mp.re(mp.diff(lambda b: exp_cross(H, lam, b, T), 0, p))


def exp_data(H, J, T=1, lam=1):
    H, T = fp.parameters(H, T)
    lam = mp.mpf(lam)
    b, a = mp.matrix(J, 1), mp.matrix(J, 1)
    for k in range(J):
        w = k*mp.pi/T
        scale = 1/mp.sqrt(T) if k == 0 else mp.sqrt(2/T)
        b[k] = scale*mp.re(exp_cross(H, lam, 1j*w, T))
        z = lam+1j*w
        a[k] = scale*mp.re(T if z == 0 else mp.expm1(z*T)/z)
    return b, a, mp.re(exp_cross(H, lam, lam, T))


def centered_power_cos(p, k):
    """Integral |t-1/2|^p h_k(t) dt on [0,1]."""
    p = mp.mpf(p)
    if p <= -1:
        raise ValueError('Require p > -1')
    if k % 2:
        return mp.mpf(0)
    scale = 1 if k == 0 else mp.sqrt(2)
    return scale*2*(-1)**(k//2)*mp.mpf('0.5')**(p+1)/(p+1)*mp.hyper(
        [(p+1)/2], [mp.mpf('0.5'), (p+3)/2], -(k*mp.pi)**2/16)


def abs_kernel_image(H, t):
    """Integral K_H(t,s)|s-1/2| ds, H>1/2, on [0,1]."""
    H, _ = fp.parameters(H)
    t = mp.mpf(t)
    if H == mp.mpf('0.5'):
        return abs(t-mp.mpf('0.5'))
    return H/2*((1-t)**(2*H-1)+t**(2*H-1))-(
        (1-t)**(2*H)+t**(2*H))/2+abs(t-mp.mpf('0.5'))**(2*H)


def abs_power(H, p):
    """<|t-1/2|,t^p> for nonnegative integral p on [0,1]."""
    H, _ = fp.parameters(H)
    if not isinstance(p, int) or p < 0:
        raise ValueError('p must be a nonnegative integer')
    if H == mp.mpf('0.5'):
        return (mp.mpf(p)/2+mp.mpf(2)**(-p-1))/((p+1)*(p+2))
    r = 2*H
    centered = mp.mpf(2)**(-p-r)*mp.fsum(
        mp.binomial(p,j)/(j+r+1) for j in range(0,p+1,2))
    return H/2*(mp.beta(p+1,r)+1/(p+r))-(
        mp.beta(p+1,r+1)+1/(p+r+1))/2+centered


def abs_data(H, J):
    H, _ = fp.parameters(H)
    a = mp.matrix([centered_power_cos(1,k) for k in range(J)])
    if H == mp.mpf('0.5'):
        return a.copy(), a, mp.mpf(1)/12
    u = fp.classical_power(H,2*H-1,J)
    v = fp.classical_power(H,2*H,J)
    b = mp.matrix([0 if k % 2 else H*u[k]-v[k]+centered_power_cos(2*H,k)
                   for k in range(J)])
    def moment(p):
        return (p/2+mp.mpf(2)**(-p-1))/((p+1)*(p+2))
    n2 = H*moment(2*H-1)-moment(2*H)+mp.mpf(2)**(-2*H-1)/(2*H+2)
    return b, a, n2


def enrichment_setup(H, L, T=1):
    """Whiten polynomial cross moments against the cosine prefix."""
    J = L.rows
    Q = [fp.power_rhs(H,p,J,T) for p in (1,2)]
    C = [fp.forward(L,q) for q in Q]
    P = mp.matrix([[fp.power_cross(H,p,q,T) for q in (1,2)] for p in (1,2)])
    return C, P


def enriched_projection(L, b, norm2, poly_rhs, C, P, n):
    """Projection on V_n+span(t,t^2), total dimension n+2.

    A 2x2 Schur complement isolates the extra directions. The caller chooses
    working precision; negative residuals are rejected, never clipped.
    """
    c = fp.forward(L[:n,:n],b[:n,0])
    U = [v[:n,0] for v in C]
    S = mp.matrix([[P[i,j]-mp.fdot(U[i],U[j]) for j in range(2)] for i in range(2)])
    r = mp.matrix([poly_rhs[i]-mp.fdot(U[i],c) for i in range(2)])
    d = mp.lu_solve(S,r)
    base2 = norm2-mp.fdot(c,c)
    e2 = base2-mp.fdot(r,d)
    eig = mp.eigsy(S,eigvals_only=True)
    if e2 <= 0 or eig[0] <= 0:
        raise ArithmeticError('Nonpositive error or Schur complement; increase precision/audit.')
    return dict(error2=e2,base_error2=base2,schur_cond=eig[1]/eig[0],
                schur_min_eigenvalue=eig[0],cancellation_digits=mp.log10(norm2/e2),
                polynomial_coefficients=d)


def endpoint_upper2(H, n, third_boundary_sum, fourth_L1, T=1):
    """Sufficient error bound for a C^4 function after quadratic correction."""
    H,T = fp.parameters(H,T)
    M = mp.mpf(third_boundary_sum)+mp.mpf(fourth_L1)
    C = mp.gamma(2*H+1)*mp.sin(mp.pi*H)
    return 2*C/T*M*M*(T/mp.pi)**(2*H+7)*mp.zeta(2*H+7,n)
