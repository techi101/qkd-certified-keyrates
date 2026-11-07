"""Portable NPZ ridge model; loads arrays with allow_pickle=False."""
import hashlib
from pathlib import Path
import numpy as np
from learn import Ridge, Quad


def save_model(model,path):
    np.savez_compressed(path,W=model.W,mu=model.q.mu,sd=model.q.sd,alpha=np.array(model.a))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path):
    with np.load(path,allow_pickle=False) as a:
        r=Ridge(float(a['alpha'])); r.W=a['W']; r.q=Quad(); r.q.mu=a['mu']; r.q.sd=a['sd']
    return r
