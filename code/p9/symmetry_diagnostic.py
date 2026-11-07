"""Development-only audit of the published C4 symmetry reduction.

Uses only three historical training points. Does not alter Protocol 4 kernels.
Seven traceless constraint coordinates; certificates lifted to 31 source moments.
Numerical tolerance checks here are not exact symmetry/security proofs.
"""
import json,time
from pathlib import Path
import numpy as np
from qkd import Model,grad_ld
from learn import herm_to_vec,split_pred
from model_store import load_model
from physical_moments import sample_inputs
from solver_audit import solve

def real_rows(A):
    return np.c_[A.real.reshape(len(A),-1),A.imag.reshape(len(A),-1)]

class CyclicModel(Model):
    def __init__(self,Nc):
        super().__init__(Nc);self.full_model=Model(Nc);self.source_G=self.G.copy();self.source_flat=self.Gflat.copy()
        nb=Nc+1;F=np.exp(2j*np.pi*np.outer(np.arange(4),np.arange(4))/4)/2
        self.T=np.kron(F,np.eye(nb));charges=np.array([(n-k)%4 for k in range(4) for n in range(nb)])
        self.mask=charges[:,None]==charges[None,:]
        ops=np.array([self.twirl(a) for a in self.source_G]);traces=np.trace(ops,axis1=1,axis2=2).real/self.dim
        centered=ops-traces[:,None,None]*np.eye(self.dim)
        U,S,_=np.linalg.svd(real_rows(centered),full_matrices=False);rank=int((S>1e-10).sum())
        assert rank==7,rank
        self.G=np.einsum('jr,jab->rab',U[:,:rank]/S[:rank],centered)
        self.Gflat=self.G.reshape(rank,-1);self.keep=list(range(rank))
        # Reconstruct every reduced operator in the original source-operator span.
        generators=np.r_[np.eye(self.dim)[None],self.source_G]
        self.lift=np.linalg.lstsq(real_rows(generators).T,real_rows(self.G).T,rcond=1e-12)[0]
        reconstruction=np.einsum('jr,jab->rab',self.lift,generators)
        self.lift_error=float(np.abs(reconstruction-self.G).max())
        assert self.lift_error<1e-12,self.lift_error

    def twirl(self,A):
        B=self.T.conj().T@A@self.T;B[~self.mask]=0
        return self.T@B@self.T.conj().T

    def source_gamma(self,L,xi,al):
        # Model.gamma uses self.keep, so call on a full model for this diagnostic.
        return self.full_model.gamma(L,xi,al)

    def gamma(self,L,xi,al):
        source=self.source_gamma(L,xi,al)
        return self.lift[0]+self.lift[1:].T@source

    def cert(self,G,lam,gam):
        # Both source and reduced formulae agree; use the unreduced pencil.
        source_lam=self.lift[1:]@lam
        offset=self.lift[0]@lam
        return float(lam@gam-offset+np.linalg.eigvalsh(G-np.tensordot(source_lam,self.source_G,1))[0])

class TwirledStart:
    def __init__(self,model,m):self.model=model;self.m=m;self.full=Model(m.Nc)
    def predict(self,X):
        # Interface expected by solver_audit/learn.split_pred: 7 tail coordinates.
        rows=[]
        for row in self.model.predict(X):
            H,_=split_pred(self.full,row)
            rows.append(np.r_[herm_to_vec(self.m.twirl(H)),np.zeros(7)])
        return np.asarray(rows)

def main():
    points=json.loads(Path('train_points.json').read_text())['train']
    full=Model(8);reduced=CyclicModel(8);rid=load_model('training_export/ridge_model.npz');symrid=TwirledStart(rid,reduced)
    rng=np.random.default_rng(801);result=[]
    # Warm-up each implementation on a development point only.
    p=points[0][:3]
    for m in [full,reduced]:solve(m,m.gamma(*p),np.zeros((m.dim,m.dim)))
    for index in [0,1,4]:
        p=points[index][:3];x=sample_inputs(*p,None,None);runs=[]
        for rep in range(3):
            methods={}
            for name in rng.permutation(['full_cold','full_ridge','cyclic_cold','cyclic_ridge']):
                m=reduced if name.startswith('cyclic') else full
                model=(symrid if name.startswith('cyclic') else rid) if name.endswith('ridge') else None
                gam=m.gamma(*p);r=solve(m,gam,np.zeros((m.dim,m.dim)),model=model,x=x)
                st=r['state'];w=st['w'];prob=np.exp(w-w.max());prob/=prob.sum();rho=(st['V']*prob)@st['V'].conj().T
                residual=np.max(np.abs(np.real(full.Gflat@rho.T.ravel())-full.gamma(*p)))
                methods[name]={'elapsed':r['elapsed_seconds'],'final_certificate':r['trajectory'][-1]['cert'],
                               'final_objective':r['trajectory'][-1]['approx_objective'],'source_residual':float(residual)}
                if residual>1e-7:raise RuntimeError((name,'source residual',residual))
            spread=np.ptp([r['final_objective'] for r in methods.values()])
            assert spread<3e-6,spread
            runs.append(methods)
        result.append({'training_index':index,'params':p,'runs':runs})
    out={'scope':'development diagnostic on 3 training points, not confirmation',
         'reduced_coordinate_count':7,'source_operator_reconstruction_error':reduced.lift_error,'results':result}
    Path('SYMMETRY_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2))
    for row in result:
        print(row['training_index'],{n:round(float(np.median([r[n]['elapsed'] for r in row['runs']])),4) for n in row['runs'][0]})

if __name__=='__main__':main()
