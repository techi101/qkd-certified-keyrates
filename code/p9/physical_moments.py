"""Exact sufficient-statistic sampler for the assumed Gaussian heterodyne model.

For each of four equally frequent prepared symbols: Z=X+iY, independent real
Gaussian quadratures, E[Z]=sqrt(eta)*i**x*alpha, E|Z-EZ|^2=1+eta*xi/2.
The observables are sqrt(2)X, sqrt(2)Y, |Z|^2-1, 2 Re(Z^2), divided by 4.
Fixed N samples per symbol; Alice's probabilities are known, not estimated.
This simulates input noise, not a finite-size confidence-region security proof.
"""
import numpy as np
from qkd import exp_sim


def detector_parameters(L, xi, alpha):
    eta = 10 ** (-.02 * L)
    mu = np.sqrt(eta) * alpha * (1j ** np.arange(4))
    variance = (1 + eta * xi / 2) / 2
    return mu, variance


def sample_inputs(L, xi, alpha, N, rng, batches=None):
    if N is None:
        return np.r_[exp_sim(L, xi, alpha).ravel(), alpha]
    if N < 2 or int(N) != N:
        raise ValueError('N must be an integer >=2, per prepared symbol')
    mu, variance = detector_parameters(L, xi, alpha)
    shape = (4,) if batches is None else (batches, 4)
    mx = rng.normal(mu.real, np.sqrt(variance / N), size=shape)
    my = rng.normal(mu.imag, np.sqrt(variance / N), size=shape)
    # Sample mean and centred sum of squares are independent for normal data.
    vx = variance * rng.chisquare(N - 1, size=shape) / N
    vy = variance * rng.chisquare(N - 1, size=shape) / N
    ex2, ey2 = vx + mx**2, vy + my**2
    moments = np.stack([np.sqrt(2)*mx, np.sqrt(2)*my,
                        ex2 + ey2 - 1, 2*(ex2 - ey2)], axis=-1) / 4
    if batches is None:
        return np.r_[moments.ravel(), alpha]
    return np.c_[moments.reshape(batches, 16), np.full(batches, alpha)]


def conditional_covariance(L, xi, alpha, symbol):
    mu, s = detector_parameters(L, xi, alpha); a, b = mu[symbol].real, mu[symbol].imag
    cov = np.diag([2*s, 2*s, 4*s*(a*a+b*b)+4*s*s, 16*s*(a*a+b*b)+16*s*s])
    cov[0,2] = cov[2,0] = 2*np.sqrt(2)*a*s
    cov[1,2] = cov[2,1] = 2*np.sqrt(2)*b*s
    cov[0,3] = cov[3,0] = 4*np.sqrt(2)*a*s
    cov[1,3] = cov[3,1] = -4*np.sqrt(2)*b*s
    cov[2,3] = cov[3,2] = 8*s*(a*a-b*b)
    return cov
