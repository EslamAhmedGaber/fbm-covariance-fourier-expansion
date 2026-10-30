"""Paper 2: deterministic integrand projection, using arbitrary precision.

The cosine Gram formulas are inherited from Paper 1 / archived hfc.py.
Power moments and power-cosine coefficients are derived in the startup note.
No negative residual is silently clipped. Precision agreement is a diagnostic,
not an interval-arithmetic certificate. Inputs should be decimal strings.
"""
import mpmath as mp


def parameters(H, T=1):
    H, T = mp.mpf(H), mp.mpf(T)
    if not mp.mpf('0.5') <= H < 1 or T <= 0:
        raise ValueError('Require 1/2 <= H < 1 and T > 0.')
    return H, T


def power_cross(H, p, q, T=1):
    """<t^p,t^q>_H; both powers are required to belong to the space."""
    H, T = parameters(H, T)
    p, q = mp.mpf(p), mp.mpf(q)
    if p <= -H or q <= -H:
        raise ValueError('A power t^p is admissible only when p > -H.')
    if H == mp.mpf('0.5'):
        return T ** (p+q+1) / (p+q+1)
    beta = 2*H-1
    return H*beta*T**(p+q+2*H)*(mp.beta(p+1, beta)+mp.beta(q+1, beta))/(p+q+2*H)


def polynomial_norm2(H, coefficients, T=1):
    H, T = parameters(H, T)
    a = list(map(mp.mpf, coefficients))
    return mp.fsum(a[i]*a[j]*power_cross(H, i, j, T)
                   for i in range(len(a)) for j in range(len(a)))


def interval_cross(H, a, b, c, d):
    """Inner product of 1_[a,b] and 1_[c,d], by increment covariance."""
    H, _ = parameters(H)
    a,b,c,d = map(mp.mpf, (a,b,c,d))
    if not 0 <= a <= b or not 0 <= c <= d:
        raise ValueError('Require nonnegative ordered endpoints.')
    return (abs(b-c)**(2*H)+abs(a-d)**(2*H)
            -abs(b-d)**(2*H)-abs(a-c)**(2*H))/2


def gram(H, J, T=1):
    """Arbitrary-precision transcription of Paper 1's cosine Gram formula."""
    H, T = parameters(H, T)
    if not isinstance(J, int) or J < 1:
        raise ValueError('J must be a positive integer.')
    if H == mp.mpf('0.5'):
        return mp.eye(J)
    beta = 2*H-1
    m = [mp.hyp1f1(beta, beta+1, 1j*mp.pi*k) for k in range(J)]
    C = 2*H*T**beta
    G = mp.matrix(J)
    G[0,0] = T**beta
    for k in range(1, J):
        G[k,k] = C*(mp.re(m[k])-2*(1-H)*mp.im(m[k])/(k*mp.pi))
        if k % 2 == 0:
            G[0,k] = G[k,0] = -mp.sqrt(2)*C*mp.im(m[k])/(k*mp.pi)
        for l in range(1, k):
            if (k+l) % 2 == 0:
                G[k,l] = G[l,k] = 2*C/mp.pi*(l*mp.im(m[l])-k*mp.im(m[k]))/(k*k-l*l)
    return G


def classical_power(H, p, J, T=1):
    """Classical cosine integrals; for p <= -1/2 these are not L2 projections."""
    H,T = parameters(H,T)
    p = mp.mpf(p)
    if p <= -H:
        raise ValueError('Require p > -H.')
    a = mp.matrix(J,1)
    a[0] = T**(p+mp.mpf('0.5'))/(p+1)
    for k in range(1,J):
        a[k] = mp.sqrt(2)*T**(p+mp.mpf('0.5'))/(p+1)*mp.hyper(
            [(p+1)/2], [mp.mpf('0.5'), (p+3)/2], -(k*mp.pi)**2/4)
    return a


def power_rhs(H, p, J, T=1):
    """b_k=<t^p,h_k>_H via 1F2 and 2F3; derived from mixed moments."""
    H,T = parameters(H,T)
    p = mp.mpf(p)
    if p <= -H:
        raise ValueError('Require p > -H.')
    if H == mp.mpf('0.5'):
        return classical_power(H,p,J,T)
    beta, z = 2*H-1, (p+2*H)/2
    b = mp.matrix(J,1)
    b[0] = power_cross(H,p,0,T)/mp.sqrt(T)
    for k in range(1,J):
        x = -(k*mp.pi)**2/4
        A = mp.beta(p+1,beta)*mp.hyper([z],[mp.mpf('0.5'),z+1],x)
        B = mp.hyper([1,z],[H,H+mp.mpf('0.5'),z+1],x)/beta
        b[k] = mp.sqrt(2)*H*beta*T**(p+2*H-mp.mpf('0.5'))/(p+2*H)*(A+B)
    return b


def _e1(H, t, w):
    return t**(2*H)/(2*H)*mp.hyp1f1(2*H,2*H+1,1j*w*t)


def indicator_rhs(H, tau, J, T=1):
    H,T = parameters(H,T)
    tau = mp.mpf(tau)
    if not 0 <= tau <= T:
        raise ValueError('Require 0 <= tau <= T.')
    a = mp.matrix(J,1)
    a[0] = tau/mp.sqrt(T)
    for k in range(1,J):
        a[k] = mp.sqrt(2/T)*T/(k*mp.pi)*mp.sin(k*mp.pi*tau/T)
    if H == mp.mpf('0.5'):
        return a,a.copy()
    b = mp.matrix(J,1)
    for k in range(J):
        w = k*mp.pi/T
        c = 1/mp.sqrt(T) if k == 0 else mp.sqrt(2/T)
        b[k] = c*H*mp.re(_e1(H,T,w)+mp.exp(1j*w*tau)*(
            mp.conj(_e1(H,tau,w))-_e1(H,T-tau,w)))
    return b,a


def forward(L,b):
    c = mp.matrix(len(b),1)
    for i in range(len(b)):
        c[i] = (b[i]-mp.fsum(L[i,j]*c[j] for j in range(i)))/L[i,i]
    return c


def backward_transpose(L,c):
    a = mp.matrix(len(c),1)
    for i in reversed(range(len(c))):
        a[i] = (c[i]-mp.fsum(L[j,i]*a[j] for j in range(i+1,len(c))))/L[i,i]
    return a


def projection_rows(H, G, L, b, a_classical, norm2, Js):
    c = forward(L,b)
    rows=[]
    for J in Js:
        cj, aj, bj = c[:J,0], a_classical[:J,0], b[:J,0]
        Gj = G[:J,:J]
        opt2 = norm2-mp.fsum(x*x for x in cj)
        classical2 = norm2-2*mp.fdot(aj,bj)+mp.fdot(aj,Gj*aj)
        if opt2 <= 0 or classical2 <= 0:
            raise ArithmeticError('Nonpositive residual: increase precision or audit inputs; no clipping.')
        rows.append({'J':J,'error2_optimal':opt2,'error2_classical':classical2,
                     'relative_optimal':mp.sqrt(opt2/norm2),
                     'relative_classical':mp.sqrt(classical2/norm2),
                     'rms_gain':mp.sqrt(classical2/opt2),
                     'cancellation_digits':mp.log10(norm2/opt2)})
    return rows


def linear_upper2(H,J,T=1):
    """Analytical bound for f(t)=t, inherited weighted cosine inequality."""
    H,T=parameters(H,T)
    if J < 1:
        raise ValueError('J must be positive.')
    s=2*H+3
    first=J if J % 2 else J+1
    tail=2**(-s)*mp.zeta(s,mp.mpf(first)/2)
    return 8*mp.gamma(2*H+1)*mp.sin(mp.pi*H)*T**(2*H+2)/mp.pi**s*tail


def indicator_upper2(H,J,T=1):
    """Uniform-in-tau analytical bound, not a floating-point enclosure."""
    H,T=parameters(H,T)
    if J < 1:
        raise ValueError('J must be positive.')
    s=2*H+1
    return 2*mp.gamma(2*H+1)*mp.sin(mp.pi*H)*T**(2*H)/mp.pi**s*mp.zeta(s,J)
