"""Test T-W2: the openQKDsecurity v2.2.0 final-step rule, float vs exact, for asymptotic QPSK DM CV-QKD (Nc = 8).

Float side reproduces, in numpy, openQKDsecurity v2.2.0 (tag commit f952c35):
  FW2StepSolver.m L246-249      rho = perturbationChannel(rho, perturbationChannelEpsilon(rho, 0, perturbationCheck=false))
  step2Solver.m  L54-74         safeCut = 1e-14 * max(lmax(G rho), lmax(Z G rho)); eps = max(epsG, epsZ) with
                                perturbationChannelEpsilon(., safeCut, step2Enhancement=true)
  primalfep.m                   fval = tr(G_eps (logmsafe(G_eps) - logmsafe(ZG_eps)))
  primalDfep.m  L448-459        gradf = G^dag( D_eps(logmsafe(G_eps)) - D_eps(logmsafe(ZG_eps)) )   ("second perturbation")
  logmsafe.m    L14-20          eig, eigenvalues < safeCut replaced by safeCut
  step2Solver.m L44, L50        beta = dual + fval - tr(gradf rho);  lowerBound = beta / (1 - eps)
with G(rho) = V rho V^dag, V = [q_0; q_1; q_2; q_3] (144 x 36), q_z = I_4 (x) sqrt(R_z), and Z the pinching on the
4-dim key register (block diagonal 36 x 36 blocks), as in the Lin et al. (2019) formulation (d' = 144).

Exact side: the SAME expression, for the SAME dyadic candidate rho (the float matrix after the first perturbation)
and the SAME eps (a binary float, taken exactly), with exact logarithms (no floor), enclosed in Arb.
36 x 36 reductions (exact; the value does not depend on the choice of factor K_z, because a change of factor is a
block-diagonal unitary on the output, which commutes with Z, with D_eps and with the logarithm):
  t = tr rho, c = eps t / 144,  A = (1-eps) rho + c I_36,  B_z = (1-eps) K_z^dag rho K_z + c I_36
  G_eps = V A V^dag + c (I - V V^dag)         -> tr log G_eps = tr log A + 108 log c
  Z G_eps = (+)_z B_z                          -> tr log ZG_eps = sum_z tr log B_z
  gradf = (1-eps) [log A - sum_z K_z log(B_z) K_z^dag] + eps (tr log G_eps - tr log ZG_eps)/144 * I_36
  fval  = tr(A log A) + 108 c log c - sum_z tr(B_z log B_z)
  value = [lambda.gamma + lambda_min(gradf - sum_i lambda_i Gamma_i) + fval - tr(rho gradf)] / (1 - eps)
Exact factors: K_z = C_z = W^z C_0 W^-z (ball Cholesky, from t1_standard_float / fastcheck, unmodified).
lambda_min: lower end by an LDL^dag shift test (ball_pd), upper end by an enclosed Rayleigh quotient at the float
minimising eigenvector; outward rounded with lo()/up().

DEVIATIONS from openQKDsecurity v2.2.0 (all deliberate; listed so the protocol can cite them):
 D1. Dual: v2 solves a CVX SDP (submaxproblem, with tolBase = 1e-15 representation slack, L117-143) and uses
     cvx_optval. We use lambda.gamma + lambda_min(gradf - sum lambda Gamma) at the GIVEN solver multipliers (the
     trace-dual form, Temesi Eq. 35). Valid in exact arithmetic for any lambda; an SDP would choose y from the
     float gradient.
 D2. Units: v2 divides by log(2) (bits); we work in nats throughout (bits = nats / ln 2; same comparison).
 D3. MATLAB eig vs numpy eigh (LAPACK zheevd); MATLAB min(eig)/max(eig) vs numpy eigvalsh.
 D4. Kraus operator: v2 takes the user's G; we use V = [I(x)sqrt(R_z)]_z with sqrt(R_z) from numpy eigh
     (Model.q) in float, Cholesky factors in the exact side (factor-independent, see above).
 D5. No constraint objects / tolerance terms (tolEq etc.); constraints are the 31 Gamma_i of the paper.
 D6. Candidate rho = the solver's state sigma (log-domain eigenpairs, exponentiated in float), not a Frank-Wolfe
     step-1 output.
 D7. Exact side encloses the stated expression including the second perturbation D_eps(log .); that step is
     part of v2's computation (primalDfep L451-453) and is reproduced, not endorsed.
"""
import os, sys, time
import numpy as np

P10_SRC = os.environ.get('P10_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                  'research_07nov_qkd_p10', 'src'))
sys.path.insert(0, os.path.abspath(P10_SRC))
from flint import arb, acb, acb_mat, ctx               # noqa: E402
import t1_standard_float as T1                         # noqa: E402  (builds Model(8), FastChecker2(8,400), Cz)
from fastcheck import to_acb, herm, eye, ball_pd, lo, up   # noqa: E402

m, fc, Cz = T1.m, T1.fc, T1.Cz
D, DP = m.dim, 4 * m.dim                                # 36, 144
Vf = np.vstack(m.q)                                     # 144 x 36 float isometry, blocks q_z = I(x)sqrt(R_z)

# ---------------------------------------------------------------- float side (openQKDsecurity v2 rule)
def _h(a): return (a + a.conj().T) / 2

def pert_channel(rho, p):                               # perturbationChannel.m L352-354
    dim = rho.shape[0]
    r = (1 - p) * rho + p * np.real(np.trace(rho)) * np.eye(dim) / dim
    return _h(r)

def pert_eps(rho, min_eig, step2=False):                # perturbationChannelEpsilon.m L386-405
    dim = rho.shape[0]; eig_min = np.linalg.eigvalsh(rho).min(); eps = 0.0
    if eig_min <= min_eig:
        if step2 and eig_min < 0:
            eig_min = 0.0
        eps = dim * (min_eig - eig_min) / (np.real(np.trace(rho)) - eig_min * dim)
    return eps

def logmsafe(A, cut):                                   # logmsafe.m L14-20
    w, U = np.linalg.eigh(_h(A)); w = np.where(w < cut, cut, w)
    return _h((U * np.log(w)) @ U.conj().T)

def G_map(r): return _h(Vf @ r @ Vf.conj().T)

def ZG_map(r):
    out = np.zeros((DP, DP), complex)
    for z, q in enumerate(m.q):
        out[z * D:(z + 1) * D, z * D:(z + 1) * D] = q @ r @ q
    return _h(out)

def float_v2(sigma, lam, gam, floor=True, eps_override=None):
    eps0 = pert_eps(sigma, 0.0)                                         # FW2StepSolver L246 (check off)
    rho = pert_channel(sigma, eps0)                                     # L247
    g, zg = G_map(rho), ZG_map(rho)
    top = max(np.linalg.eigvalsh(g).max(), np.linalg.eigvalsh(zg).max())
    safecut = top * 1e-14                                               # step2Solver L67-68
    eps = max(pert_eps(g, safecut, True), pert_eps(zg, safecut, True))  # L70-73
    if eps_override is not None:
        eps = eps_override
    cut = safecut if floor else 1e-300
    ge, zge = pert_channel(g, eps), pert_channel(zg, eps)
    LG, LZ = logmsafe(ge, cut), logmsafe(zge, cut)
    fval = float(np.real(np.trace(ge @ (LG - LZ))))                     # primalfep (nats)
    grad = _h(Vf.conj().T @ (pert_channel(LG, eps) - pert_channel(LZ, eps)) @ Vf)   # primalDfep L452-459
    dual = float(lam @ gam + np.linalg.eigvalsh(grad - np.tensordot(lam, m.G, 1))[0])   # D1
    beta = dual + fval - float(np.real(np.trace(grad @ rho)))           # step2Solver L44
    return dict(value=beta / (1 - eps), eps0=eps0, eps=eps, safecut=safecut, rho=rho, dual=dual, fval=fval)

# ---------------------------------------------------------------- exact side (Arb)
def _tr(M):
    return sum((M[i, i].real for i in range(M.nrows())), arb(0))

def exact_v2(rho_f, eps_f, lam, params, prec):
    ctx.prec = prec
    rho = herm(to_acb(rho_f)); e = arb(float(eps_f)); one = arb(1)
    t = _tr(rho); c = e * t / DP
    if not c > 0:
        return {'status': 'c_not_positive'}
    A = herm(rho * (one - e) + eye(D) * c)
    LA = T1.log_herm(A, prec)
    trLG = _tr(LA) + (DP - D) * c.log()
    fval = _tr(A * LA) + (DP - D) * c * c.log()
    H = acb_mat(D, D); trLZ = arb(0)
    for C in Cz:
        Ch = C.conjugate().transpose()
        Bz = herm(Ch * rho * C * (one - e) + eye(D) * c)
        Lz = T1.log_herm(Bz, prec)
        H = H + C * Lz * Ch; trLZ += _tr(Lz); fval -= _tr(Bz * Lz)
    grad = herm((LA - H) * (one - e) + eye(D) * (e * (trLG - trLZ) / DP))
    trrg = _tr(rho * grad)
    pencil = grad
    for l, gm in zip(lam, fc.Gam):
        pencil = pencil - gm * arb(float(l))
    pencil = herm(pencil)
    mid = np.array([[complex(float(pencil[i, j].real.mid()), float(pencil[i, j].imag.mid())) for j in range(D)]
                    for i in range(D)])
    wv, vv = np.linalg.eigh(_h(mid)); tmin = float(wv[0]); s = 1e-12 * max(1, abs(tmin))
    for _ in range(8):
        if ball_pd(pencil - eye(D) * arb(tmin - s)):
            break
        s *= 100
    else:
        return {'status': 'lmin_not_proved'}
    v = to_acb(vv[:, :1]); vh = v.conjugate().transpose()
    rq = (vh * pencil * v)[0, 0].real / (vh * v)[0, 0].real
    gam = fc.gamma(*params)
    lin = sum((arb(float(l)) * g for l, g in zip(lam, gam)), arb(0))
    rest = fval - trrg
    return {'status': 'ok', 'exact_lo': lo((lin + arb(tmin - s) + rest) / (one - e)),
            'exact_hi': up((lin + rq + rest) / (one - e)), 'prec': prec}

def sigma_from(w, V):
    lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
    return _h((V * np.exp(lw)) @ V.conj().T)

def tw2_state(state_w, state_V, lam, params, floor=True, eps_override=None):
    """Returns dict(float_value, exact_lo, exact_hi, eps_used, safecut, status, ...)."""
    w, V, lam = np.asarray(state_w, float), np.asarray(state_V, complex), np.asarray(lam, float)
    t0 = time.perf_counter()
    gam = m.gamma(*params)
    fl = float_v2(sigma_from(w, V), lam, gam, floor, eps_override)
    t1 = time.perf_counter(); r = {'status': 'none'}
    for prec in (400, 800):
        try:
            r = exact_v2(fl['rho'], fl['eps'], lam, params, prec)
            if r['status'] == 'ok':
                break
        except ValueError as exc:
            r = {'status': 'eig_fail:' + str(exc)[:60]}
    out = dict(float_value=fl['value'], exact_lo=r.get('exact_lo'), exact_hi=r.get('exact_hi'),
               eps_used=fl['eps'], eps0=fl['eps0'], safecut=fl['safecut'], status=r['status'],
               prec=r.get('prec'), floor=floor, float_s=t1 - t0, exact_s=time.perf_counter() - t1)
    if r['status'] == 'ok':
        out['float_minus_hi'] = fl['value'] - r['exact_hi']
    return out
