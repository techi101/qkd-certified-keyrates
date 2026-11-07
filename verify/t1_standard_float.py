"""Trial 1: does the STANDARD dense-matrix float certificate (logm by eigh, eps-perturbation
as in Winick et al. 2018 / Model.grad + Model.regular) overstate the exact certificate of the
SAME candidate matrix?  Candidate S = regular(sigma_dense, eps) is a float matrix, taken as an
exact dyadic matrix.  We enclose  lin + lambda_min(log S - sum_z K_z log(K_z^+ S K_z) K_z^+ - sum l Gam)
from both sides in Arb:  lower end by LDL shift test, upper end by an enclosed Rayleigh quotient.
If float_cert > upper end, the float certificate is PROVEN to exceed the exact value."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from flint import arb, acb, acb_mat, ctx
from common import load_states
from qkd import Model
from fastcheck import FastChecker2, to_acb, herm, eye, ball_pd, lo, up

m = Model(8)
fc = FastChecker2(8, 400)   # fixed operators enclosed at 1600 bits; Cholesky factor C0, W powers
n = fc.n
# C_z = W^z C0 W^-z factors I_A (x) R_z (verified symmetry lemma); eig-free
Cz = [fc.Wp[z] * fc.C0 * fc.Wph[z] for z in range(4)]

def log_herm(S, prec):
    for alg in ('vdhoeven_mourrain', 'rump'):
        try:
            E, L, R = S.eig(left=True, right=True, algorithm=alg)
            D = acb_mat(S.nrows(), S.nrows())
            for i, e in enumerate(E):
                if not e.real > 0: raise ValueError('eigenvalue not provably positive')
                D[i, i] = acb(e.real).log()
            return R * D * L
        except ValueError as exc:
            last = exc
    raise ValueError(str(last))

def exact_bound(Sf, lam, params, prec):
    ctx.prec = prec
    S = herm(to_acb(Sf))
    if not ball_pd(S): return {'status': 'S_not_PD'}
    grad = log_herm(S, prec)
    for C in Cz:
        Ch = C.conjugate().transpose()
        grad = grad - C * log_herm(herm(Ch * S * C), prec) * Ch
    pencil = grad
    for l, gm in zip(lam, fc.Gam): pencil = pencil - gm * arb(float(l))
    pencil = herm(pencil)
    mid = np.array([[complex(float(pencil[i, j].real.mid()), float(pencil[i, j].imag.mid())) for j in range(n)] for i in range(n)])
    wv, vv = np.linalg.eigh((mid + mid.conj().T) / 2); t = float(wv[0])
    shift = 1e-12 * max(1, abs(t))
    for _ in range(8):
        if ball_pd(pencil - eye(n) * arb(t - shift)): break
        shift *= 100
    else: return {'status': 'lmin_not_proved'}
    v = to_acb(vv[:, :1]); vh = v.conjugate().transpose()
    rq = (vh * pencil * v)[0, 0].real / (vh * v)[0, 0].real
    gam = fc.gamma(*params)
    lin = sum((arb(float(l)) * g for l, g in zip(lam, gam)), arb(0))
    return {'status': 'ok', 'lo': lo(lin + arb(t - shift)), 'hi': up(lin + rq)}

if __name__ == '__main__':
    idx = [int(x) for x in sys.argv[1].split(',')]
    epss = [float(x) for x in sys.argv[2].split(',')]
    S = load_states(); out = []
    for i in idx:
        st, rec = S[i]['st'], S[i]['rec']
        w, V, lam = st['w'], st['V'], st['lambda']
        lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
        sig = (V * np.exp(lw)) @ V.conj().T; sig = (sig + sig.conj().T) / 2
        gam = m.gamma(*rec['params'])
        for eps in epss:
            Sf = m.regular(sig, eps) if eps > 0 else sig
            fcert = float(m.cert(m.grad(Sf), lam, gam))
            t = time.perf_counter(); r = {'status': 'none'}
            for prec in (400, 800):
                try:
                    r = exact_bound(Sf, lam, rec['params'], prec); r['prec'] = prec; break
                except ValueError as exc:
                    r = {'status': 'eig_fail:' + str(exc)[:60]}
            r.update(i=i, cohort=rec['cohort'], eps=eps, float_std=fcert, A_routeA=rec['A'], s=time.perf_counter() - t,
                     leak=rec['leak_bits'])
            if r['status'] == 'ok':
                r['float_minus_hi'] = fcert - r['hi']; r['float_minus_lo'] = fcert - r['lo']
            out.append(r)
            print(json.dumps(r), flush=True)
    json.dump(out, open(sys.argv[3], 'w'), indent=1)
