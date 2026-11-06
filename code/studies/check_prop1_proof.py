"""Independent check of the self-contained proof of Proposition 1 (closed cosine Gram).
Direct quadrature of H*b*int int h_k(u) h_l(v) |u-v|^(b-1) du dv (singularity removed analytically
by the substitution x = u - v) against the closed formula, at (H, T) not used elsewhere in the paper,
plus the intermediate identity G_kk + S_kk used in the proof."""
import mpmath as mp
mp.mp.dps = 30
def M(b, x): return mp.hyp1f1(b, b + 1, 1j * x)
def h(k, T):
    return (lambda t: 1 / mp.sqrt(T)) if k == 0 else (lambda t: mp.sqrt(2 / T) * mp.cos(k * mp.pi * t / T))
def direct(H, T, k, l):
    b = 2 * H - 1
    hk, hl = h(k, T), h(l, T)
    # int_0^T int_0^T hk(u) hl(v) |u-v|^(b-1) = int_0^T x^(b-1) [ int_0^{T-x} (hk(v+x) hl(v) + hk(v) hl(v+x)) dv ] dx
    g = lambda x: mp.quad(lambda v: hk(v + x) * hl(v) + hk(v) * hl(v + x), [0, T - x])
    # x^(b-1) singular at 0: substitute x = y^(1/b) -> dx = (1/b) y^(1/b - 1) dy, x^(b-1) dx = dy / b
    val = mp.quad(lambda y: g(y ** (1 / b)) / b, [0, T ** b])
    return H * b * val
def closed(H, T, k, l):
    b = 2 * H - 1; cT = 2 * H * T ** b
    m = lambda j: M(b, j * mp.pi)
    if (k + l) % 2: return mp.mpf(0)
    if k == 0 and l == 0: return T ** b
    if k == 0 or l == 0:
        j = max(k, l); return -mp.sqrt(2) * cT * m(j).imag / (mp.pi * j)
    if k == l: return cT * (m(k).real - (1 - b) * m(k).imag / (k * mp.pi))
    return (2 * cT / mp.pi) * (l * m(l).imag - k * m(k).imag) / (k ** 2 - l ** 2)
H, T = mp.mpf('0.7'), mp.mpf('1.7')
worst = 0
for (k, l) in [(0, 0), (0, 2), (0, 6), (1, 1), (2, 2), (5, 5), (1, 3), (2, 4), (3, 7), (2, 3), (4, 10)]:
    d, c = direct(H, T, k, l), closed(H, T, k, l)
    worst = max(worst, abs(d - c))
    print(f"(k,l)=({k},{l})  direct={mp.nstr(d, 16)}  closed={mp.nstr(c, 16)}  diff={mp.nstr(abs(d - c), 3)}")
# identity used in the proof: Re M_{b+1}(k pi)/(b+1) = -r_k/(k pi)
b = 2 * H - 1
for k in (1, 4, 9):
    print('identity', k, mp.nstr(M(b + 1, k * mp.pi).real / (b + 1) + M(b, k * mp.pi).imag / (k * mp.pi), 3))
print('max |direct - closed| =', mp.nstr(worst, 3))
