"""Ball-arithmetic (python-flint/Arb) final certificate for the cutoff model.

Every quantity is an enclosure: if all steps succeed, the true value of
sum(lambda*gamma) + lambda_min(grad f(sigma) - sum(lambda*Gamma)) for the
exactly defined sigma = exp(H)/tr exp(H) lies above the returned lower bound.
Scope: finite cutoff model only. Not a statement about the cutoff itself.
"""
import time
import numpy as np
from flint import arb, acb, acb_mat, ctx
from qkd import HERM4, DROP


def to_acb(a):
    a = np.asarray(a, dtype=complex)
    return acb_mat([[acb(float(v.real), float(v.imag)) for v in row] for row in a])


def kron(a, b):
    ra, ca, rb, cb = a.nrows(), a.ncols(), b.nrows(), b.ncols()
    out = acb_mat(ra * rb, ca * cb)
    for i in range(ra):
        for j in range(ca):
            x = a[i, j]
            if x == 0:
                continue
            for k in range(rb):
                for l in range(cb):
                    out[i * rb + k, j * cb + l] = x * b[k, l]
    return out


def herm(a):
    return (a + a.conjugate().transpose()) * arb(0.5)


def eye(n):
    m = acb_mat(n, n)
    for i in range(n):
        m[i, i] = 1
    return m


def lo(x):
    """Float that is guaranteed <= the lower end of ball x (directed rounding)."""
    return float(np.nextafter(float(x.lower()), -np.inf))


def up(x):
    return float(np.nextafter(float(x.upper()), np.inf))


def ball_cholesky(A):
    """Lower-triangular C with C C^dag = A, computed in ball arithmetic.
    Raises unless every pivot is provably positive."""
    n = A.nrows()
    C = acb_mat(n, n)
    for j in range(n):
        s = A[j, j].real
        for k in range(j):
            s -= (C[j, k] * C[j, k].conjugate()).real
        if not s > 0:
            raise ValueError('Cholesky pivot not provably positive')
        d = s.sqrt()
        C[j, j] = d
        for i in range(j + 1, n):
            v = A[i, j]
            for k in range(j):
                v -= C[i, k] * C[j, k].conjugate()
            C[i, j] = v / d
    return C


def herm_fun(a, fun):
    """f(A) for Hermitian A from a rigorous diagonalisation L A R = D (Arb eig).
    Raises if eigenvalues are not isolated (then nothing is claimed)."""
    E, L, R = a.eig(left=True, right=True, algorithm='rump')
    n = a.nrows()
    D = acb_mat(n, n)
    for i, e in enumerate(E):
        D[i, i] = fun(acb(e.real))
    return R * D * L, E


class FastChecker:
    def __init__(self, Nc, prec=400):
        self.Nc, self.prec = Nc, prec
        ctx.prec = prec
        nb = Nc + 1
        pi = arb.pi()
        R = []
        for z in range(3):
            r = acb_mat(nb, nb)
            for n in range(nb):
                r[n, n] = arb(1) / 4
                for k in range(n + 1, nb):
                    d = n - k
                    sn = arb(0) if d % 4 == 0 else (arb(-1) ** (d // 4)) * (1 if d % 2 == 0 else 1 / arb(2).sqrt())
                    ang = 2 * (acb(0, 1) ** ((z * d) % 4)) * sn / d
                    rad = (arb(n + k) / 2 + 1).gamma() / (2 * pi * (arb(n + 1).gamma() * arb(k + 1).gamma()).sqrt())
                    r[n, k] = ang * rad
                    r[k, n] = r[n, k].conjugate()
            R.append(r)
        R.append(eye(nb) - R[0] - R[1] - R[2])
        self.Rops = R
        self.sqrtR =[kron(eye(4), herm_fun(r, lambda x: x.sqrt())[0]) for r in R]
        a = acb_mat(nb, nb)
        for n in range(1, nb):
            a[n - 1, n] = arb(n).sqrt()
        ad = a.conjugate().transpose()
        s2 = arb(2).sqrt()
        diagn = acb_mat(nb, nb)
        for n in range(nb):
            diagn[n, n] = n
        ops = [(ad + a) / s2, acb(0, 1) * (ad - a) / s2, diagn, a * a + ad * ad]
        proj = []
        for x in range(4):
            p = acb_mat(4, 4); p[x, x] = 1; proj.append(p)
        g = [kron(p, o) for p in proj for o in ops] + [kron(to_acb(b), eye(nb)) for b in HERM4]
        self.Gam = [v for i, v in enumerate(g) if i != DROP]

    def gamma(self, L, xi, al):
        L, xi, al = arb(L), arb(xi), arb(al)
        eta = arb(10) ** (-L / 50)
        I = acb(0, 1)
        vals = []
        for x in range(4):
            c = (I ** x) * al
            vals += [(2 * eta).sqrt() * c.real / 4, (2 * eta).sqrt() * c.imag / 4,
                     (eta * al ** 2 + eta * xi / 2) / 4, 2 * eta * (-1) ** x * al ** 2 / 4]
        alice = [[(-(al ** 2) * (1 - I ** (i - j))).exp() / 4 for j in range(4)] for i in range(4)]
        for b in HERM4:
            vals.append(sum((alice[i][j] * acb(complex(b[j, i])) for i in range(4) for j in range(4)), acb(0)).real)
        return [v for i, v in enumerate(vals) if i != DROP]

    def check(self, state, params):
        t0 = time.perf_counter()
        ctx.prec = self.prec
        w, V, lam = state['w'], state['V'], state['lambda']
        lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
        # sigma is DEFINED as exp(H)/tr exp(H) with H the exactly representable float matrix below.
        H = herm(to_acb((V * lw) @ V.conj().T))
        eH = H.exp()
        Z = eH.trace()
        logsig = H - eye(H.nrows()) * Z.log()
        sigma = herm(eH / Z)
        grad = logsig
        for q in self.sqrtR:
            Lb, _ = herm_fun(herm(q * sigma * q), lambda x: x.log())
            grad = grad - q * Lb * q
        pencil = grad
        for l, gm in zip(lam, self.Gam):
            pencil = pencil - gm * arb(float(l))
        pencil = herm(pencil)
        E = pencil.eig(algorithm='rump')
        lmin = min((e.real for e in E), key=lambda x: float(x.mid()))
        lower_eig = arb(min(float(e.real.lower()) for e in E))
        gam = self.gamma(*params)
        lin = sum((arb(float(l)) * gv for l, gv in zip(lam, gam)), arb(0))
        bound = lin + lower_eig
        return {'certified_lower_bound': float(bound.lower()), 'midpoint': float(bound.mid()),
                'radius': float(bound.rad()), 'prec_bits': self.prec,
                'seconds': time.perf_counter() - t0}


def ball_pd(M):
    """Rigorous positive-definiteness test of a Hermitian ball matrix by LDL^H in Arb.
    Returns True only if every pivot is provably positive."""
    n = M.nrows()
    A = [[M[i, j] for j in range(n)] for i in range(n)]
    for k in range(n):
        d = A[k][k].real
        if not d > 0:
            return False
        for i in range(k + 1, n):
            f = A[i][k] / d
            if f == 0:
                continue
            for j in range(k + 1, i + 1):
                A[i][j] -= f * A[j][k].conjugate()
    return True


class FastChecker2(FastChecker):
    """Same certificate, cheaper: sigma is defined from the C4-twirled H (exact twirl),
    so log(q_z sigma q_z) = W^z log(q_0 sigma q_0) W^-z holds exactly and one
    eigendecomposition replaces four; exp(H) is taken per symmetry sector;
    lambda_min is bounded by a float shift proved by a ball LDL test."""

    def __init__(self, Nc, prec=400, alg='vdhoeven_mourrain'):
        # Fixed operators are enclosed once at the top of the precision ladder,
        # so a retry at higher precision is not limited by their radii.
        super().__init__(Nc, 4 * prec)
        self.prec = prec
        self.alg = alg
        nb = Nc + 1
        self.n = 4 * nb
        I = acb(0, 1)
        F = acb_mat([[(I ** ((j * k) % 4)) / 2 for k in range(4)] for j in range(4)])
        self.T = kron(F, eye(nb))
        self.Th = self.T.conjugate().transpose()
        ch = [(n - k) % 4 for k in range(4) for n in range(nb)]
        self.sectors = [[i for i in range(self.n) if ch[i] == c] for c in range(4)]
        shift = acb_mat(4, 4)
        for i in range(4):
            shift[(i + 1) % 4, i] = 1
        D = acb_mat(nb, nb)
        for k in range(nb):
            D[k, k] = I ** (k % 4)
        W = kron(shift, D)
        self.Wp = [eye(self.n)]
        for _ in range(3):
            self.Wp.append(W * self.Wp[-1])
        self.Wph = [w.conjugate().transpose() for w in self.Wp]
        self.q0 = self.sqrtR[0]
        # Any factor with C C^dag = R_0 gives the same objective and gradient
        # (S(C^dag rho C) = S(rho^1/2 R rho^1/2)); a ball Cholesky factor needs no eig.
        # C_z = D^z C D^-z factors R_z = D^z R_0 D^-z, so the W-covariance still holds.
        self.C0 = kron(eye(4), ball_cholesky(self.Rops[0]))
        self.C0h = self.C0.conjugate().transpose()

    def check(self, state, params):
        """Whole pipeline is recomputed at 400, 800, 1600 bits until certification succeeds."""
        t0 = time.perf_counter()
        last = None
        w = state['w']
        lmin = float((w - (w.max() + np.log(np.exp(w - w.max()).sum()))).min())
        # Very graded states (smallest log-eigenvalue below -140) skip the 400-bit attempt.
        ladder = [(self.prec, self.alg), (2 * self.prec, self.alg), (2 * self.prec, 'rump'),
                  (4 * self.prec, self.alg), (4 * self.prec, 'rump')]
        if lmin < -140:
            ladder = ladder[1:]
        for prec, alg in ladder:
            try:
                out = self._check_at(state, params, prec, alg)
                out['seconds'] = time.perf_counter() - t0
                return out
            except ValueError as exc:
                last = exc
        raise ValueError('not certified on the 400/800/1600-bit ladder: ' + str(last))

    def _check_at(self, state, params, prec, alg):
        t0 = time.perf_counter()
        ctx.prec = prec
        w, V, lam = state['w'], state['V'], state['lambda']
        lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
        Hp = self.Th * herm(to_acb((V * lw) @ V.conj().T)) * self.T
        n = self.n
        Hs = acb_mat(n, n); E = acb_mat(n, n)
        for ix in self.sectors:  # exact twirl: keep only sector blocks
            blk = acb_mat([[Hp[i, j] for j in ix] for i in ix])
            blk = herm(blk)
            eb = blk.exp()
            for a, i in enumerate(ix):
                for b, j in enumerate(ix):
                    Hs[i, j] = blk[a, b]; E[i, j] = eb[a, b]
        Z = E.trace()
        logsig = self.T * Hs * self.Th - eye(n) * Z.log()
        sigma = herm(self.T * E * self.Th / Z)
        B0 = herm(self.C0h * sigma * self.C0)
        Ev = None
        try:
            Ev, L, R = B0.eig(left=True, right=True, algorithm=alg)
            self.used = (prec, alg)
        except ValueError:
            pass
        if Ev is None:
            raise ValueError('eigenvalues of q0 sigma q0 not isolated at any tried precision')
        Dg = acb_mat(n, n)
        for i, e in enumerate(Ev):
            Dg[i, i] = acb(e.real).log()
        K0 = self.C0 * (R * Dg * L) * self.C0h
        grad = logsig
        for wz, wzh in zip(self.Wp, self.Wph):
            grad = grad - wz * K0 * wzh
        pencil = grad
        for l, gm in zip(lam, self.Gam):
            pencil = pencil - gm * arb(float(l))
        pencil = herm(pencil)
        mid = np.array([[complex(pencil[i, j].real.mid()) + 1j * float(pencil[i, j].imag.mid()) for j in range(n)] for i in range(n)])
        t = float(np.linalg.eigvalsh((mid + mid.conj().T) / 2)[0])
        shift = 1e-12 * max(1.0, abs(t))
        ok = False
        for _ in range(6):
            tt = t - shift
            if ball_pd(pencil - eye(n) * arb(tt)):
                ok = True; break
            shift *= 100
        if not ok:
            raise ValueError('could not prove lambda_min bound')
        gam = self.gamma(*params)
        lin = sum((arb(float(l)) * gv for l, gv in zip(lam, gam)), arb(0))
        bound = lin + arb(tt)
        return {'certified_lower_bound': lo(bound), 'certified_upper_end': up(bound), 'shift': shift, 'prec_bits': prec,
                'seconds': time.perf_counter() - t0}


class LoewnerCrossCheck(FastChecker2):
    """Independent second route for the one step that used Arb's (experimental) eig.
    Take ANY Hermitian point matrix U (here the midpoint of the eig-based log).
    Prove B0 <= exp(c) exp(U) by a ball-LDL positive-definiteness test, using only
    Arb's matrix exponential and arithmetic. Operator monotonicity of log then gives
    log B0 <= U + c I, so replacing log B0 by U + c I can only LOWER the bound.
    The eig result is used only to choose U; correctness does not depend on it."""

    def cross(self, state, params, prec=800, c=1e-9):
        t0 = time.perf_counter()
        ctx.prec = prec
        w, V, lam = state['w'], state['V'], state['lambda']
        lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
        Hp = self.Th * herm(to_acb((V * lw) @ V.conj().T)) * self.T
        n = self.n
        Hs = acb_mat(n, n); E = acb_mat(n, n)
        for ix in self.sectors:
            blk = herm(acb_mat([[Hp[i, j] for j in ix] for i in ix]))
            eb = blk.exp()
            for a, i in enumerate(ix):
                for b, j in enumerate(ix):
                    Hs[i, j] = blk[a, b]; E[i, j] = eb[a, b]
        Z = E.trace()
        logsig = self.T * Hs * self.Th - eye(n) * Z.log()
        sigma = herm(self.T * E * self.Th / Z)
        B0 = herm(self.C0h * sigma * self.C0)
        Ev, L, R = None, None, None
        for alg in (self.alg, 'rump'):
            try:
                Ev, L, R = B0.eig(left=True, right=True, algorithm=alg); break
            except ValueError:
                continue
        if Ev is None:
            raise ValueError('could not obtain a candidate U')
        Dg = acb_mat(n, n)
        for i, e in enumerate(Ev):
            Dg[i, i] = acb(e.real).log()
        Lb = R * Dg * L
        # U: exact Hermitian point matrix (midpoints), origin irrelevant to rigour.
        U = acb_mat(n, n)
        for i in range(n):
            for j in range(n):
                U[i, j] = acb(Lb[i, j].real.mid(), Lb[i, j].imag.mid())
        U = herm(U)
        U = acb_mat([[acb(U[i, j].real.mid(), U[i, j].imag.mid()) for j in range(n)] for i in range(n)])
        if not ball_pd(herm(U.exp() * arb(c).exp() - B0)):
            raise ValueError('Loewner test B0 <= e^c e^U not proved')
        K0 = self.C0 * (U + eye(n) * arb(c)) * self.C0h
        grad = logsig
        for wz, wzh in zip(self.Wp, self.Wph):
            grad = grad - wz * K0 * wzh
        pencil = grad
        for l, gm in zip(lam, self.Gam):
            pencil = pencil - gm * arb(float(l))
        pencil = herm(pencil)
        mid = np.array([[complex(float(pencil[i, j].real.mid()), float(pencil[i, j].imag.mid())) for j in range(n)] for i in range(n)])
        t = float(np.linalg.eigvalsh((mid + mid.conj().T) / 2)[0])
        shift = 1e-12 * max(1.0, abs(t))
        for _ in range(6):
            tt = t - shift
            if ball_pd(pencil - eye(n) * arb(tt)):
                break
            shift *= 100
        else:
            raise ValueError('lambda_min bound not proved')
        gam = self.gamma(*params)
        lin = sum((arb(float(l)) * gv for l, gv in zip(lam, gam)), arb(0))
        return {'cross_lower_bound': lo(lin + arb(tt)), 'c': c, 'prec': prec,
                'seconds': time.perf_counter() - t0}
