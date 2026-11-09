"""Ball-arithmetic enclosure of the EC leakage delta_EC = H(Z) - beta*I(X;Z) (bits).

Same model as research_02nov_verifier/src/keyrate.py, but every quantity is an
Arb ball: the sector integrals use certified integration (acb.integral), so the
net key rate  A/ln2 - delta_EC  becomes a fully enclosed number.
beta = 95/100 exactly; L, xi, alpha are the exact binary floats of the protocol points.
"""
import sys, json, time
from flint import arb, acb, ctx

def sector_prob(mu, nbar, z):
    s2 = (1 + nbar) / 2
    rot = acb(0, -1) ** z  # e^{-i z pi/2} exactly
    c = rot * mu
    a, b = c.real, c.imag
    k = (arb.pi() * s2 / 2).sqrt()
    def f(t, analytic):
        u = a * t.cos() + b * t.sin()
        v2 = a * a + b * b - u * u
        g = (-v2 / (2 * s2)).exp() / (2 * arb.pi() * s2)
        return g * (s2 * (-(u * u) / (2 * s2)).exp() + u * k * (1 + (u / (2 * s2).sqrt()).erf()))
    q = arb.pi() / 4
    return acb.integral(f, -q, q).real

def xlog2x(p):
    return p * p.log() / arb(2).log()

def leakage_bits(L, xi, alpha, beta=arb(95) / 100):
    L, xi, alpha = arb(L), arb(xi), arb(alpha)
    eta = arb(10) ** (-arb(2) * L / 100)
    nbar = eta * xi / 2
    P = [[sector_prob(eta.sqrt() * alpha * acb(0, 1) ** x, nbar, z) for z in range(4)] for x in range(4)]
    pz = [sum((P[x][z] for x in range(4)), arb(0)) / 4 for z in range(4)]
    HZ = -sum((xlog2x(p) for p in pz), arb(0))
    HZX = -sum((xlog2x(P[x][z]) for x in range(4) for z in range(4)), arb(0)) / 4
    rowsum = [sum(P[x], arb(0)) for x in range(4)]
    return HZ - beta * (HZ - HZX), rowsum

if __name__ == '__main__':
    ctx.prec = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    d = json.load(open(sys.argv[1]))
    out = []
    t0 = time.time()
    for r in d['res']:
        t = time.time()
        leak, rs = leakage_bits(*r['params'])
        A = arb(r['A'])  # A is the certified lower end (already outward-rounded float)
        rate = A / arb(2).log() - leak
        out.append(dict(params=r['params'], cohort=r['cohort'], leak_mid=float(leak.mid()), leak_rad=float(leak.rad()),
                        leak_float=r['leak_bits'], rate_lower=float(rate.lower()), rate_float=r['rate_bits_float'],
                        rate_cert_prev=r['rate_bits_certified'], rowsum_ok=all(arb(1) in s for s in rs), s=time.time() - t))
        print(len(out), '%.3e' % out[-1]['leak_rad'], '%.2e' % (out[-1]['leak_mid'] - r['leak_bits']), out[-1]['rowsum_ok'], '%.2fs' % out[-1]['s'], flush=True)
    json.dump(dict(prec=ctx.prec, total_s=time.time() - t0, res=out), open('enclosed_leak_out.json', 'w'), indent=1)
