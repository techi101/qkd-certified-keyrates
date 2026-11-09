"""Published reflection/C4 reductions and block Newton: numerical model only.
Reference: Temesi et al., arXiv:2608.23285v2 Appendices I/J.
No new symmetry theorem and no exact/interval certificate claimed.
"""
import time
import numpy as np
from scipy.linalg import block_diag
from qkd import Model, _graded_logsq, grad_ld
from symmetry_diagnostic import CyclicModel, real_rows
from learn import split_pred

class ReflectedModel(CyclicModel):
    def __init__(self, Nc):
        super().__init__(Nc)
        nb=Nc+1
        ra=np.eye(4)[[0,3,2,1]]
        self.J=np.kron(ra,np.eye(nb))
        ops=np.array([self.project(a) for a in self.source_G])
        traces=np.trace(ops,axis1=1,axis2=2).real/self.dim
        centered=ops-traces[:,None,None]*np.eye(self.dim)
        u,s,_=np.linalg.svd(real_rows(centered),full_matrices=False)
        rank=int((s>1e-10).sum())
        if rank!=6:raise RuntimeError(('Reflection rank',rank))
        self.G=np.einsum('jr,jab->rab',u[:,:rank]/s[:rank],centered)
        self.Gflat=self.G.reshape(rank,-1);self.keep=list(range(rank))
        generators=np.r_[np.eye(self.dim)[None],self.source_G]
        self.lift=np.linalg.lstsq(real_rows(generators).T,real_rows(self.G).T,rcond=1e-12)[0]
        rec=np.einsum('jr,jab->rab',self.lift,generators)
        self.lift_error=float(np.abs(rec-self.G).max())
        if self.lift_error>1e-12:raise RuntimeError(('Source lift',self.lift_error))
        charges=np.array([(n-k)%4 for k in range(4) for n in range(nb)])
        self.indices=[np.where(charges==q)[0] for q in range(4)]
        self.Gsector=self.T.conj().T@self.G@self.T
        self.Gblocks=[self.Gsector[:,ix[:,None],ix] for ix in self.indices]
        self.q0=self.q[0]
        shift=np.roll(np.eye(4),1,axis=0)
        self.W=np.kron(shift,np.diag((1j)**np.arange(nb)))
        self.Wpowers=[np.linalg.matrix_power(self.W,z) for z in range(4)]
        # Verify covariance before using one output-channel factorisation.
        self.channel_covariance_error=max(float(np.abs(w@self.q0@w.conj().T-self.q[z]).max()) for z,w in enumerate(self.Wpowers))
        if self.channel_covariance_error>1e-12:raise RuntimeError(('Channel covariance',self.channel_covariance_error))

    def reflect(self,A):return self.J@A.conj()@self.J.conj().T

    def project(self,A):
        a=self.twirl(A)
        return (a+self.reflect(a))/2

    def eigh(self,H):
        h=self.T.conj().T@H@self.T
        off=float(np.abs(np.where(self.mask,0,h)).max())
        if off>1e-8:raise RuntimeError(('Noninvariant Hamiltonian',off))
        values=[];basis=np.zeros_like(h)
        for ix in self.indices:
            w,v=np.linalg.eigh(h[np.ix_(ix,ix)])
            values.append(w);basis[np.ix_(ix,ix)]=v
        # One global partition function, never independent block normalization.
        return np.concatenate(values),self.T@basis[:,np.concatenate(self.indices)]

    def eigenvalues(self,H):
        h=self.T.conj().T@H@self.T
        if float(np.abs(np.where(self.mask,0,h)).max())>1e-8:raise RuntimeError('Noninvariant line-search Hamiltonian')
        return np.concatenate([np.linalg.eigvalsh(h[np.ix_(ix,ix)]) for ix in self.indices])

    def gradient(self,w,V):
        lw=w-(w.max()+np.log(np.exp(w-w.max()).sum()))
        logsigma=(V*lw)@V.conj().T
        C=V*np.exp(lw/2)
        log0,ls=_graded_logsq(self.q0@C)
        h0=self.q0@log0@self.q0
        H=sum(a@h0@a.conj().T for a in self.Wpowers)
        H=(H+H.conj().T)/2
        # All four output channels have the same spectrum by verified C4 covariance.
        F=float(np.exp(lw)@lw-4*np.exp(ls)@ls)
        G=logsigma-H
        return (G+G.conj().T)/2,H,F


def project_newton(m,H,gam,lam,tol=1e-9,maxit=50):
    H=m.project(H)
    def evaluate(l, vectors=True):
        A=H-np.tensordot(l,m.G,1)
        if vectors:w,V=m.eigh(A)
        else:w=m.eigenvalues(A);V=None
        mx=w.max();p=np.exp(w-mx);Z=p.sum();p/=Z
        return float(np.log(Z)+mx+l@gam),w,V,p
    g,w,V,p=evaluate(lam);steps=0;backtracks=0
    for _ in range(maxit):
        Mt=V.conj().T@m.G@V
        means=np.real(np.diagonal(Mt,axis1=1,axis2=2))@p;gr=gam-means
        if np.abs(gr).max()<tol:break
        dw=w[:,None]-w[None,:];dp=p[:,None]-p[None,:];big=np.abs(dw)>1e-12
        K=np.where(big,dp/np.where(big,dw,1),p[:,None])
        flat=Mt.reshape(len(Mt),-1)
        Hess=np.real(flat.conj()@(flat*K.ravel()).T)-np.outer(means,means)
        direction=-np.linalg.lstsq(Hess,gr,rcond=1e-12)[0];step=1.
        while True:
            trial,*_=evaluate(lam+step*direction,False)
            if trial<=g+1e-4*step*(gr@direction) or step<1e-6:break
            step/=2;backtracks+=1
        lam=lam+step*direction;g,w,V,p=evaluate(lam);steps+=1
    rho=(V*p)@V.conj().T
    residual=float(np.abs(np.real(m.Gflat@rho.T.ravel())-gam).max())
    return w,V,lam,{'residual':residual,'trace_error':float(abs(np.trace(rho)-1)),
                   'newton_steps':steps,'line_search_rejections':backtracks,'inner_converged':residual<tol}


def solve_optimized(m,gam,initial,model=None,x=None,budget=30.):
    start=time.perf_counter();infer=0.
    if model is not None:
        # Reuse the same frozen full-space predictor. Projection is paid by inference.
        row=model.predict(np.atleast_2d(x))[0]
        initial,_=split_pred(m.full_model,row)
        initial=m.project(initial);infer=time.perf_counter()-start
    w,V,lam,diag=project_newton(m,initial,gam,np.zeros(len(m.keep)))
    traj=[];best=-np.inf;failures=0
    while True:
        G,H,F=m.gradient(w,V)
        # Source-coordinate full pencil deliberately retained for a common final
        # spectral certificate; do not infer exact security from float arithmetic.
        cert=float(m.cert(G,-lam,gam));best=max(best,cert)
        failures+=int(not diag['inner_converged'])
        row={'t':time.perf_counter()-start,'cert':cert,'approx_objective':F,**diag};traj.append(row)
        if F-best<=1e-6 or row['t']>budget:
            lw=w-(w.max()+np.log(np.exp(w-w.max()).sum()))
            return {'trajectory':traj,'inference_seconds':infer,'elapsed_seconds':time.perf_counter()-start,
                    'failed_inner_steps':failures,'stopped_at_target':bool(F-best<=1e-6),
                    'state':{'w':w,'V':V,'lambda':-lam,'log_sigma':(V*lw)@V.conj().T}}
        w,V,lam,diag=project_newton(m,H,gam,lam)
