"""Error-correction leakage for QPSK reverse reconciliation under the simulated channel.

p(z|x): probability that the heterodyne outcome of the displaced thermal state
(mean sqrt(eta)*alpha*i^x, mean thermal photon number eta*xi/2, as in the
moment model of qkd.py) falls in key sector z = [z*pi/2 - pi/4, z*pi/2 + pi/4).
Outcome density (Husimi Q): exp(-|b-mu|^2/(1+n))/(pi(1+n)).
delta_EC = H(Z) - beta*I(X;Z) in bits (uniform X, p_pass = 1).
Floating point; this is a reporting quantity, not part of the certificate.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import erf


def sector_prob(mu, nbar, z):
    """P(arg b in sector z) for b ~ complex Gaussian mean mu, E|b-mu|^2 = 1+nbar."""
    s2 = (1 + nbar) / 2  # variance per real quadrature
    c = np.exp(-1j * z * np.pi / 2) * mu  # rotate so the sector is centred on angle 0
    a, b = c.real, c.imag

    def radial(t):
        # integral over r >= 0 of the Gaussian density along direction t, times r
        u = a * np.cos(t) + b * np.sin(t)
        v2 = a * a + b * b - u * u
        g = np.exp(-v2 / (2 * s2)) / (2 * np.pi * s2)
        return g * (s2 * np.exp(-u * u / (2 * s2)) + u * np.sqrt(np.pi * s2 / 2) * (1 + erf(u / np.sqrt(2 * s2))))

    return quad(radial, -np.pi / 4, np.pi / 4, epsabs=1e-13, epsrel=1e-12)[0]


def leakage_bits(L, xi, alpha, beta=0.95):
    eta = 10 ** (-0.2 * L / 10)
    nbar = eta * xi / 2
    P = np.array([[sector_prob(np.sqrt(eta) * alpha * 1j ** x, nbar, z) for z in range(4)] for x in range(4)])
    P /= P.sum(1, keepdims=True)
    pz = P.mean(0)
    h = lambda p: -np.sum(p[p > 0] * np.log2(p[p > 0]))
    HZ = h(pz)
    HZX = np.mean([h(P[x]) for x in range(4)])
    return HZ - beta * (HZ - HZX), P


if __name__ == '__main__':
    d, P = leakage_bits(20, .02, .7)
    print('rows sum', P.sum(1), 'leak bits %.6f' % d)
