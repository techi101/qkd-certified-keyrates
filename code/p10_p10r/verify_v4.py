"""High-precision diagnostic of the final cutoff-model certificate.

Closed-form operators are rebuilt at adaptive precision. This is mpmath
recomputation with a Cholesky margin, not interval/rational certification.
"""
import time
from math import ceil
import numpy as np
import mpmath as mp
from qkd import HERM4,DROP

def matrix(a):
    return mp.matrix([[mp.mpc(complex(v)) for v in row] for row in np.asarray(a)])

def kron(a,b):
    return mp.matrix([[a[i//b.rows,j//b.cols]*b[i%b.rows,j%b.cols]
                      for j in range(a.cols*b.cols)] for i in range(a.rows*b.rows)])

def herm_function(a,fun):
    e,q=mp.eigh((a+a.H)/2)
    if min(e)<=0:raise ValueError('Nonpositive eigenvalue in high-precision matrix function')
    return q*mp.diag([fun(v) for v in e])*q.H

class Checker:
    def __init__(self,Nc):self.Nc=Nc;self.cache={}

    def setup(self,dps):
        mp.mp.dps=dps
        if dps in self.cache:return self.cache[dps]
        nb=self.Nc+1;R=[];imag=mp.mpc(0,1)
        for z in range(3):
            r=mp.matrix(nb,nb)
            for n in range(nb):
                r[n,n]=mp.mpf(1)/4
                for k in range(n+1,nb):
                    diff=n-k
                    sn=mp.mpf(0) if diff%4==0 else mp.mpf(-1)**(diff//4)*(1 if diff%2==0 else 1/mp.sqrt(2))
                    angle=2*imag**((z*diff)%4)*sn/diff
                    radial=mp.gamma(1+mp.mpf(n+k)/2)/(2*mp.pi*mp.sqrt(mp.factorial(n)*mp.factorial(k)))
                    r[n,k]=angle*radial;r[k,n]=mp.conj(r[n,k])
            R.append(r)
        R.append(mp.eye(nb)-sum(R,mp.matrix(nb,nb)))
        sqrt_ops=[kron(mp.eye(4),herm_function(r,mp.sqrt)) for r in R]
        a=mp.matrix(nb,nb)
        for n in range(1,nb):a[n-1,n]=mp.sqrt(n)
        ops=[(a.H+a)/mp.sqrt(2),imag*(a.H-a)/mp.sqrt(2),mp.diag(list(range(nb))),a*a+a.H*a.H]
        projectors=[]
        for x in range(4):
            p=mp.matrix(4,4);p[x,x]=1;projectors.append(p)
        g=[kron(p,o) for p in projectors for o in ops]+[kron(matrix(b),mp.eye(nb)) for b in HERM4]
        g=[v for i,v in enumerate(g) if i!=DROP]
        self.cache[dps]=(sqrt_ops,g)
        return sqrt_ops,g

    def gamma(self,L,xi,al):
        L,xi,al=map(mp.mpf,(L,xi,al));eta=mp.mpf(10)**(-L/50);imag=mp.mpc(0,1);values=[]
        for x in range(4):
            c=imag**x*al
            values += [mp.sqrt(2*eta)*mp.re(c)/4,mp.sqrt(2*eta)*mp.im(c)/4,
                       (eta*al**2+eta*xi/2)/4,2*eta*(-1)**x*al**2/4]
        alice=mp.matrix([[mp.exp(-al**2*(1-imag**(i-j)))/4 for j in range(4)] for i in range(4)])
        for b in HERM4:
            bm=matrix(b);values.append(mp.re(sum(alice[i,j]*bm[j,i] for i in range(4) for j in range(4))))
        return [v for i,v in enumerate(values) if i!=DROP]

    def check(self,state,params,float_cert):
        start=time.perf_counter();w,V,lam=state['w'],state['V'],state['lambda']
        lw=w-(w.max()+np.log(np.exp(w-w.max()).sum()))
        dps=int(10*ceil((-float(lw.min())/np.log(10)+30)/10))
        sqrt_ops,gam_ops=self.setup(dps);gam=self.gamma(*params)
        vh=matrix(V);sigma=vh*mp.diag([mp.exp(mp.mpf(float(v))) for v in lw])*vh.H
        sigma=(sigma+sigma.H)/2;sigma/=sum(sigma[i,i] for i in range(sigma.rows))
        gradient=herm_function(sigma,mp.log)
        for q in sqrt_ops:gradient-=q*herm_function(q*sigma*q,mp.log)*q
        pencil=gradient-sum((mp.mpf(float(l))*g for l,g in zip(lam,gam_ops)),mp.matrix(sigma.rows,sigma.cols))
        pencil=(pencil+pencil.H)/2
        lower=float(np.linalg.eigvalsh(np.array(pencil.tolist(),dtype=complex))[0])-1e-12
        mp.cholesky(pencil-mp.mpf(lower)*mp.eye(sigma.rows))
        value=float(sum(mp.mpf(float(l))*g for l,g in zip(lam,gam))+mp.mpf(lower))
        return {'certificate_recomputed':value,'float_certificate':float_cert,
                'difference_nats':value-float_cert,'dps':dps,
                'verification_seconds_including_setup':time.perf_counter()-start,
                'kind':'high-precision numerical diagnostic; not interval-certified'}
