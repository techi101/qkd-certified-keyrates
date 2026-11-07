"""v2 learning: observable-moment inputs (fix 4), cheap Gibbs labels (fix 8), ridge and MLP predictors (fix 14)."""
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from qkd import exp_sim, run_gibbs_ld, grad_ld

def raw_inputs(L, xi, al, noiseN=None, rng=None):
    es = exp_sim(L, xi, al)
    if noiseN:  # finite-sample mean noise; per-moment sd approximated by sqrt(1 + 2<n>) / sqrt(N), / 4 for p(x)
        nbar = 4 * es[:, 2:3]
        es = es + rng.normal(size=es.shape) * np.sqrt(1 + 2 * nbar) / np.sqrt(noiseN) / 4
    return np.concatenate([es.ravel(), [al]])

def herm_to_vec(M):
    iu = np.triu_indices(M.shape[0]); iu1 = np.triu_indices(M.shape[0], 1)
    return np.concatenate([M[iu].real, M[iu1].imag])

def vec_to_herm(v, d):
    iu = np.triu_indices(d); iu1 = np.triu_indices(d, 1); n0 = len(iu[0])
    M = np.zeros((d, d), complex); M[iu] = v[:n0]; M[iu1] += 1j * v[n0:]
    return M + np.triu(M, 1).conj().T

def make_label(m, L, xi, al, budget=60.0):
    """Gibbs run to a 1e-6-nat gap; label = (log sigma*, certificate multipliers)."""
    gam = m.gamma(L, xi, al); t0 = time.perf_counter()
    traj, (w, V), lam = run_gibbs_ld(m, gam, np.zeros((m.dim, m.dim)), budget)
    lw = w - (w.max() + np.log(np.exp(w - w.max()).sum()))
    return herm_to_vec((V * lw) @ V.conj().T), -lam, time.perf_counter() - t0, traj[-1][2] - max(x[1] for x in traj)

class Quad:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12; return self
    def __call__(self, X):
        Z = (np.atleast_2d(X) - self.mu) / self.sd; n = Z.shape[1]; iu = np.triu_indices(n)
        return np.hstack([np.ones((len(Z), 1)), Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])

class Ridge:
    def __init__(self, a): self.a = a
    def fit(self, X, Y):
        self.q = Quad().fit(X); F = self.q(X)
        self.W = np.linalg.solve(F.T @ F + self.a * np.eye(F.shape[1]), F.T @ Y); return self
    def predict(self, X): return self.q(X) @ self.W

class MLP:
    def fit(self, X, Y):
        self.sx, self.sy = StandardScaler().fit(X), StandardScaler().fit(Y)
        self.net = MLPRegressor(hidden_layer_sizes=(128, 128), activation="tanh", alpha=1e-4, early_stopping=True, max_iter=5000,
                                random_state=0, n_iter_no_change=50).fit(self.sx.transform(X), self.sy.transform(Y))
        return self
    def predict(self, X):
        return self.sy.inverse_transform(self.net.predict(self.sx.transform(np.atleast_2d(X))).reshape(len(np.atleast_2d(X)), -1))

def split_pred(m, y):
    nl = len(m.keep); return vec_to_herm(y[:-nl], m.dim), y[-nl:]

def run_C(m, model, x, gam, ks=(0, 5, 10, 20, 50)):
    """Learned certificate: predicted log sigma + multipliers, then k L-BFGS steps. Rows: (time, cert, nan)."""
    t0 = time.perf_counter()
    logsig, lam0 = split_pred(m, model.predict(x)[0])
    w, V = np.linalg.eigh(logsig); G, _, _ = grad_ld(m, w, V); base = time.perf_counter() - t0; rows = []
    for k in ks:
        t1 = time.perf_counter()
        c = m.cert(G, lam0, gam) if k == 0 else m.lbfgs(lam0, G, gam, k)[1]
        rows.append((base + time.perf_counter() - t1, c, float("nan")))
    return rows

def run_B(m, model, x, gam, budget):
    t0 = time.perf_counter(); logsig, _ = split_pred(m, model.predict(x)[0])
    traj, _, _ = run_gibbs_ld(m, gam, logsig, budget, t0=t0)
    return traj


def fit_models(m, train, BUDGET, with_mlp=True):
    """Cheap Gibbs labels, ridge alpha on a fixed inner split (first 80%), then ridge (+ MLP) on all of train."""
    import time
    from qkd import run_gibbs_ld
    t0 = time.time(); Y, lab = [], []
    for p in train:
        yl, yc, tl, gap = make_label(m, *p[:3]); Y.append(np.concatenate([yl, yc])); lab.append((tl, gap))
    Y = np.array(Y); X = np.array([raw_inputs(*p[:3]) for p in train])
    print(f"labels: {len(train)} points, median {np.median([l[0] for l in lab]):.2f} s, max gap {max(l[1] for l in lab):.1e}, total {time.time()-t0:.0f} s", flush=True)
    
    # ridge alpha on a fixed inner split (first 80% of train), scored by median C(k=10) gap against the label's candidate value
    cut = int(0.8 * len(train)); Ftr = []
    for p in train[cut:]:
        tr, _, _ = run_gibbs_ld(m, m.gamma(*p[:3]), np.zeros((m.dim, m.dim)), BUDGET); Ftr.append(min(r[2] for r in tr))
    best = None
    for a in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        r = Ridge(a).fit(X[:cut], Y[:cut])
        gaps = [F - run_C(m, r, X[cut + i:cut + i + 1], m.gamma(*p[:3]), ks=(10,))[0][1] for i, (p, F) in enumerate(zip(train[cut:], Ftr))]
        if best is None or np.median(gaps) < best[0]: best = (np.median(gaps), a)
    print("ridge alpha:", best, flush=True)
    models = {"ridge": Ridge(best[1]).fit(X, Y), "mlp": MLP().fit(X, Y)} if with_mlp else {"ridge": Ridge(best[1]).fit(X, Y)}
    return models, best, lab
