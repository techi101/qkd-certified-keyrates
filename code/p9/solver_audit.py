"""Instrument the frozen log-domain solver without calling approximate F a proof.

Gibbs math/stopping parameters follow Protocol 3. Each outer iterate logs its
affine residual and trace error. Upper objectives remain explicitly approximate.
"""
import time
import numpy as np
from qkd import grad_ld


def projection(m, H, gam, lam, tol=1e-9, maxit=50):
    G = m.G
    def ev(l):
        w, V = np.linalg.eigh(H - np.tensordot(l, G, 1))
        mx = w.max(); p = np.exp(w-mx); Z = p.sum(); p /= Z
        return np.log(Z)+mx+l@gam, w, V, p
    g,w,V,p = ev(lam); steps=0
    for _ in range(maxit):
        Mt=V.conj().T@G@V
        mean=np.real(np.diagonal(Mt,axis1=1,axis2=2))@p; gr=gam-mean
        if np.abs(gr).max()<tol: break
        dw=w[:,None]-w[None,:]; dp=p[:,None]-p[None,:]; big=np.abs(dw)>1e-12
        K=np.where(big,dp/np.where(big,dw,1),p[:,None])
        Mf=Mt.reshape(len(Mt),-1)
        Hess=np.real(Mf.conj()@(Mf*K.ravel()).T)-np.outer(mean,mean)
        dl=-np.linalg.lstsq(Hess,gr,rcond=1e-12)[0]; t=1.
        while True:
            g2,w2,V2,p2=ev(lam+t*dl)
            if g2<=g+1e-4*t*(gr@dl) or t<1e-6: break
            t/=2
        lam,g,w,V,p=lam+t*dl,g2,w2,V2,p2; steps+=1
    rho=(V*p)@V.conj().T
    residual=float(np.abs(np.real(m.Gflat@rho.T.ravel())-gam).max())
    return w,V,lam,{'residual':residual,'trace_error':float(abs(np.trace(rho)-1)),
                    'newton_steps':steps,'inner_converged':residual<tol}


def solve(m, gam, initial, budget=30., model=None, x=None):
    start=time.perf_counter(); infer=0.
    if model is not None:
        y=model.predict(np.atleast_2d(x))[0]
        from learn import split_pred
        initial,_=split_pred(m,y); infer=time.perf_counter()-start
    w,V,lam,diag=projection(m,initial,gam,np.zeros(len(m.keep)))
    trajectory=[]; best=-np.inf; failed_inner=0
    while True:
        G,H,F=grad_ld(m,w,V); c=float(m.cert(G,-lam,gam)); best=max(best,c)
        failed_inner+=int(not diag['inner_converged'])
        row={'t':time.perf_counter()-start,'cert':c,'approx_objective':float(F),**diag}
        trajectory.append(row)
        if F-best<=1e-6 or row['t']>budget:
            lw=w-(w.max()+np.log(np.exp(w-w.max()).sum()))
            return {'trajectory':trajectory,'inference_seconds':infer,
                    'elapsed_seconds':time.perf_counter()-start,'failed_inner_steps':failed_inner,
                    'stopped_at_target':bool(F-best<=1e-6),
                    'state':{'w':w,'V':V,'lambda':-lam,'log_sigma':(V*lw)@V.conj().T}}
        w,V,lam,diag=projection(m,H,gam,lam)


def serialise(result):
    return {k:v for k,v in result.items() if k!='state'}
