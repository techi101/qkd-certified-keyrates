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
        self.sqrtR = [kron(eye(4), herm_fun(r, lambda x: x.sqrt())[0]) for r in R]
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
        B0 = herm(self.q0 * sigma * self.q0)
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
        K0 = self.q0 * (R * Dg * L) * self.q0
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
        return {'certified_lower_bound': float(bound.lower()), 'shift': shift, 'prec_bits': prec,
                'seconds': time.perf_counter() - t0}
