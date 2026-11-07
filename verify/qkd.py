"""v2 core: QPSK DM CV-QKD model (closed forms ported from ConicQKD.jl examples/dmcv.jl, MIT),
supported-log gradient (fix 3), Gibbs method of Temesi et al. 2608.23285, smoothed-dual L-BFGS, gap bookkeeping."""
import time
import numpy as np
from math import lgamma, pi, sqrt, exp, log
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import erf

def sinkpi4(k):
    if k % 4 == 0: return 0.0
    s = (-1.0) ** (k // 4)
    return s if k % 2 == 0 else s / sqrt(2)

def region_ops(Nc):
    R = []
    for z in range(3):
        M = np.zeros((Nc + 1, Nc + 1), complex)
        for n in range(Nc + 1):
            M[n, n] = 0.25
            for m in range(n + 1, Nc + 1):
                ang = 2 * (1j ** ((z * (n - m)) % 4)) * sinkpi4(n - m) / (n - m)
                rad = exp(lgamma(1 + (n + m) / 2) - 0.5 * (lgamma(1 + n) + lgamma(1 + m))) / (2 * pi)
                M[n, m] = ang * rad; M[m, n] = np.conj(M[n, m])
        R.append(M)
    R.append(np.eye(Nc + 1) - sum(R))
    return R

def het_ops(Nc):
    a = np.diag(np.sqrt(np.arange(1, Nc + 1)), 1).astype(complex)
    return [(a.conj().T + a) / sqrt(2), 1j * (a.conj().T - a) / sqrt(2), np.diag(np.arange(Nc + 1)).astype(complex), a @ a + a.conj().T @ a.conj().T]

E = lambda i, j: np.outer(np.eye(4)[i], np.eye(4)[j])
HERM4 = [E(i, i) for i in range(4)] + [E(i, j) + E(j, i) for i in range(4) for j in range(i + 1, 4)] \
      + [1j * (E(i, j) - E(j, i)) for i in range(4) for j in range(i + 1, 4)]
DROP = 19  # E(3,3)(x)I: implied by the trace

def exp_sim(L, xi, al):
    eta = 10 ** (-2 * L / 100); es = np.zeros((4, 4))
    for x in range(4):
        c = 1j ** x * al
        es[x] = [sqrt(2 * eta) * c.real, sqrt(2 * eta) * c.imag, eta * al ** 2 + eta * xi / 2, 2 * eta * (-1) ** x * al ** 2]
    return es / 4

def alice(al):
    r = np.zeros((4, 4), complex)
    for j in range(4):
        for i in range(j + 1):
            r[i, j] = 0.25 * np.exp(-al ** 2 * (1 - 1j ** (i - j))); r[j, i] = np.conj(r[i, j])
    return r

def p_ab(L, xi, al):
    """Joint P(x, z): radial integral in closed form, angular integral by quad."""
    eta = 10 ** (-2 * L / 100); s = 1 + eta * xi / 2; c = sqrt(eta) * al; P = np.zeros((4, 4))
    for x in range(4):
        ph = pi / 2 * x
        def inner(th):
            k = c * np.cos(th - ph)
            return np.exp(-c * c / s) * (s / 2 + k * np.exp(k * k / s) * sqrt(pi * s) / 2 * (1 + erf(k / sqrt(s))))
        for z in range(4):
            P[x, z] = quad(inner, pi * (2 * z - 1) / 4, pi * (2 * z + 1) / 4, epsabs=1e-14, epsrel=1e-13)[0]
    return P / (4 * pi * s)

def h_cond(P):  # H(Z|X) in bits from P[x, z]
    px = P.sum(1, keepdims=True); Q = P / px
    return float(-(P * np.log2(np.where(Q > 0, Q, 1))).sum())

class Model:
    def __init__(self, Nc, sup_tol=1e-12):
        self.Nc = Nc; nb = Nc + 1; self.dim = d = 4 * nb
        self.ops = het_ops(Nc); self.R = region_ops(Nc)
        Gam = [np.kron(E(x, x), self.ops[z]) for x in range(4) for z in range(4)] + [np.kron(B, np.eye(nb)) for B in HERM4]
        self.GamFull = np.array(Gam); self.keep = [i for i in range(len(Gam)) if i != DROP]
        self.G = self.GamFull[self.keep]; self.Gflat = self.G.reshape(len(self.keep), -1)
        self.q, self.W = [], []  # sqrt(A_i) and an orthonormal basis of supp(A_i) (fix 3)
        for r in self.R:
            w, V = np.linalg.eigh(r)
            self.q.append(np.kron(np.eye(4), (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T))
            self.W.append(np.kron(np.eye(4), V[:, w > sup_tol]))
        self.ker = [int((np.linalg.eigvalsh(r) <= sup_tol).sum()) for r in self.R]

    def gamma(self, L, xi, al):
        es = exp_sim(L, xi, al); A = alice(al)
        g = np.array([es[x, z] for x in range(4) for z in range(4)] + [np.trace(A @ B).real for B in HERM4])
        return g[self.keep]

    @staticmethod
    def _logm(M):
        w, V = np.linalg.eigh((M + M.conj().T) / 2); w = np.clip(w, 1e-300, None)
        return (V * np.log(w)) @ V.conj().T, w

    def H_of(self, s):  # sum_i sqrt(A_i) log_supp(sqrt(A_i) s sqrt(A_i)) sqrt(A_i)
        out = np.zeros_like(s)
        for q, W in zip(self.q, self.W):
            Y = W.conj().T @ q @ s @ q @ W; Ly, _ = self._logm(Y)
            out += q @ W @ Ly @ W.conj().T @ q
        return (out + out.conj().T) / 2

    def grad(self, s):
        Ls, _ = self._logm(s); g = Ls - self.H_of(s)
        return (g + g.conj().T) / 2

    def fval(self, s):
        ent = lambda w: -float(np.sum(w * np.log(w)))
        tot = -ent(np.clip(np.linalg.eigvalsh(s), 1e-300, None))
        for q, W in zip(self.q, self.W):
            tot += ent(np.clip(np.linalg.eigvalsh(W.conj().T @ q @ s @ q @ W), 1e-300, None))
        return tot

    def regular(self, rho, eps):
        rho = (rho + rho.conj().T) / 2; w, V = np.linalg.eigh(rho); w = np.clip(w, 0, None); w /= w.sum()
        return (1 - eps) * (V * w) @ V.conj().T + eps * np.eye(self.dim) / self.dim

    # ---- Gibbs projection (Temesi Sec. 4.2): damped Newton with the BKM Hessian
    def gibbs(self, H, gam, lam, tol=1e-10, maxit=50):
        G = self.G
        def ev(l):
            w, V = np.linalg.eigh(H - np.tensordot(l, G, 1)); m = w.max()
            p = np.exp(w - m); Z = p.sum(); p /= Z
            return np.log(Z) + m + l @ gam, w, V, p
        g, w, V, p = ev(lam)
        for _ in range(maxit):
            Mt = V.conj().T @ G @ V
            mean = np.real(np.diagonal(Mt, axis1=1, axis2=2)) @ p; gr = gam - mean
            if np.abs(gr).max() < tol: break
            dw = w[:, None] - w[None, :]; dp = p[:, None] - p[None, :]; big = np.abs(dw) > 1e-12
            K = np.where(big, dp / np.where(big, dw, 1), p[:, None])
            Mf = Mt.reshape(len(Mt), -1); Hs = np.real(Mf.conj() @ (Mf * K.ravel()).T) - np.outer(mean, mean)
            dl = -np.linalg.lstsq(Hs, gr, rcond=1e-12)[0]; t = 1.0
            while True:
                g2, w2, V2, p2 = ev(lam + t * dl)
                if g2 <= g + 1e-4 * t * (gr @ dl) or t < 1e-6: break
                t /= 2
            lam, g, w, V, p = lam + t * dl, g2, w2, V2, p2
        rho = (V * p) @ V.conj().T
        return (rho + rho.conj().T) / 2, lam

    # ---- certificates: any full-rank sigma, any multipliers (Temesi Eq. 35, sign convention lam -> -lam)
    def cert(self, G, lam, gam):
        return lam @ gam + np.linalg.eigvalsh(G - np.tensordot(lam, self.G, 1))[0]

    def smooth_neg(self, lam, G, gam, mu=1e-4):
        w, V = np.linalg.eigh(G - np.tensordot(lam, self.G, 1)); p = np.exp(-(w - w[0]) / mu); Z = p.sum()
        P = (V * (p / Z)) @ V.conj().T
        return -(lam @ gam + w[0] - mu * np.log(Z)), -(gam - np.real(self.Gflat @ P.T.ravel()))

    def lbfgs(self, lam0, G, gam, k):
        r = minimize(self.smooth_neg, lam0, args=(G, gam), jac=True, method="L-BFGS-B", options={"maxiter": k, "maxcor": 20})
        return r.x, self.cert(G, r.x, gam)

    def run_gibbs(self, gam, chi_log, budget, refine_every=0, eps=1e-6, target_gap=1e-6, t0=None, traj=None, lam0=None):
        """Temesi outer loop. Records (time, certificate, candidate F, sigma, multipliers)."""
        t0 = time.perf_counter() if t0 is None else t0; traj = [] if traj is None else traj
        sig, lam = self.gibbs(chi_log, gam, np.zeros(len(self.keep)) if lam0 is None else lam0); it = 0
        while True:
            Gc = self.grad(sig); Fc = self.fval(sig)
            Gr = Gc if eps == 0 else self.grad(self.regular(sig, eps))
            c = self.cert(Gr, -lam, gam); lam_best = -lam
            if refine_every and it % refine_every == 0:
                l2, c2 = self.lbfgs(-lam, Gr, gam, 20)
                if c2 > c: c, lam_best = c2, l2
            traj.append((time.perf_counter() - t0, c, Fc))
            it += 1
            if Fc - c <= target_gap or time.perf_counter() - t0 > budget:
                return traj, sig, lam
            sig, lam = self.gibbs(self.H_of(sig), gam, lam)  # chi = exp(log sig - grad) = exp(H(sig))


# ---- log-domain evaluation (needed when sigma has eigenvalues far below machine epsilon)
from scipy.linalg import qr as _qr, svd as _svd

def _graded_logsq(B):
    """log(B B^dagger) and its eigenvalues (in logs), accurate for column-graded B: pivoted QR, then SVD of R."""
    Q, Rm, piv = _qr(B, pivoting=True, mode="economic")
    U, s, _ = _svd(Rm, lapack_driver="gesvd")
    U = Q @ U; ls = 2 * np.log(np.maximum(s, 1e-320))
    return (U * ls) @ U.conj().T, ls

def grad_ld(m, w, V):
    """Gradient, H(sigma) and F(sigma) for sigma = V diag(exp(w)) V^dagger / Z, all in log domain."""
    lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))           # log eigenvalues of sigma
    logsig = (V * lw) @ V.conj().T
    Hs = np.zeros_like(logsig); F = float(np.exp(lw) @ lw)            # -H(sigma) = sum p log p
    C = V * np.exp(lw / 2)
    for q in m.q:
        L, ls = _graded_logsq(q @ C); Hs += q @ L @ q; F -= float(np.exp(ls) @ ls)
    Hs = (Hs + Hs.conj().T) / 2; G = logsig - Hs
    return (G + G.conj().T) / 2, Hs, F

def gibbs_ld(m, H, gam, lam, tol=1e-9, maxit=50):
    """Same Newton as Model.gibbs but returns the eigen-pair (w, V) of H - lam.M for log-domain use."""
    G = m.G
    def ev(l):
        w, V = np.linalg.eigh(H - np.tensordot(l, G, 1)); mx = w.max()
        p = np.exp(w - mx); Z = p.sum(); p /= Z
        return np.log(Z) + mx + l @ gam, w, V, p
    g, w, V, p = ev(lam)
    for _ in range(maxit):
        Mt = V.conj().T @ G @ V
        mean = np.real(np.diagonal(Mt, axis1=1, axis2=2)) @ p; gr = gam - mean
        if np.abs(gr).max() < tol: break
        dw = w[:, None] - w[None, :]; dp = p[:, None] - p[None, :]; big = np.abs(dw) > 1e-12
        K = np.where(big, dp / np.where(big, dw, 1), p[:, None])
        Mf = Mt.reshape(len(Mt), -1); Hs = np.real(Mf.conj() @ (Mf * K.ravel()).T) - np.outer(mean, mean)
        dl = -np.linalg.lstsq(Hs, gr, rcond=1e-12)[0]; t = 1.0
        while True:
            g2, w2, V2, p2 = ev(lam + t * dl)
            if g2 <= g + 1e-4 * t * (gr @ dl) or t < 1e-6: break
            t /= 2
        lam, g, w, V, p = lam + t * dl, g2, w2, V2, p2
    return w, V, lam

def run_gibbs_ld(m, gam, chi_log, budget, refine_every=0, target_gap=1e-6, t0=None, traj=None, lam0=None):
    """Temesi outer loop in log domain. traj rows: (time, certificate, candidate F)."""
    t0 = time.perf_counter() if t0 is None else t0; traj = [] if traj is None else traj
    w, V, lam = gibbs_ld(m, chi_log, gam, np.zeros(len(m.keep)) if lam0 is None else lam0); it = 0
    while True:
        Gc, Hs, Fc = grad_ld(m, w, V)
        c = m.cert(Gc, -lam, gam)
        if refine_every and it % refine_every == 0:
            c = max(c, m.lbfgs(-lam, Gc, gam, 20)[1])
        traj.append((time.perf_counter() - t0, c, Fc)); it += 1
        if Fc - max(x[1] for x in traj) <= target_gap or time.perf_counter() - t0 > budget:
            return traj, (w, V), lam
        w, V, lam = gibbs_ld(m, Hs, gam, lam)
