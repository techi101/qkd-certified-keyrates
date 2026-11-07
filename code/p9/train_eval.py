"""Train Liu et al. (2022)-style key-rate networks and audit their predictions with certified bounds.
Usage: python train_eval.py train.npz test.npz out.json [epochs]
"""
import sys, json, time
import numpy as np
import torch
from torch import nn

LN2 = np.log(2)


def rates(d, key):
    return d[key] / LN2 - d['leak']


def make_net(kind):
    if kind.startswith('liu'):
        # Table III of Liu et al. 2022 (QPSK heterodyne): 1024 sigmoid / 256 tanh / 256 sigmoid, dropout as listed
        return nn.Sequential(nn.Linear(29, 1024), nn.Sigmoid(), nn.Dropout(0.0769),
                             nn.Linear(1024, 256), nn.Tanh(), nn.Dropout(0.0362),
                             nn.Linear(256, 256), nn.Sigmoid(), nn.Dropout(0.0481), nn.Linear(256, 1))
    raise ValueError(kind)


def asym_loss(pred, y, gamma, eps):
    # Zhou et al. 2022 Eq. (3) / Liu et al. 2022 Eq. (14): e = y_pred* - y*, labels y* = -log10(rate)
    e = pred - y
    return (gamma * (e ** 2 + torch.clamp(e, min=-np.log10(eps))) - (1 - gamma) * torch.clamp(e, max=0)).mean()


def train(kind, Xtr, ytr, Xva, yva, seed, epochs, loss_kind):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net(kind); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)[:, None]
    lf = (lambda p, y: asym_loss(p, y, 0.0539, 0.8727)) if loss_kind == 'asym' else nn.functional.mse_loss
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); l = lf(net(Xt[idx]), yt[idx]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lf(net(Xv), yv))
        if v < best - 1e-7:
            best, bad = v, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(best_state); net.eval()
    return net, ep + 1, best


def main():
    tr, te = np.load(sys.argv[1]), np.load(sys.argv[2])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    torch.set_num_threads(4)
    # As in the published work: train only on instances with a positive key rate (float label).
    rtr = rates(tr, 'float_cert'); keep = tr['ok'] & (rtr > 0)
    X, y = tr['X'][keep], -np.log10(rtr[keep])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    X = (X - mu) / sd
    rng = np.random.default_rng(20261103)
    perm = rng.permutation(len(X)); nva = len(X) // 10
    va, trn = perm[:nva], perm[nva:]
    rte_f = rates(te, 'float_cert'); rte_U = rates(te, 'U'); rte_A = rates(te, 'A')
    tmask = te['ok'] & (rte_f > 0) & np.isfinite(rte_A)
    Xte = (te['X'][tmask] - mu) / sd
    out = {'n_train_used': int(len(trn)), 'n_val': int(nva), 'n_test_positive_certified': int(tmask.sum()),
           'n_test_total': int(len(rte_f)), 'n_test_uncertified': int((te['ok'] & ~np.isfinite(rte_A)).sum()), 'models': []}
    L = te['params'][tmask, 0]
    preds_all = {}
    for loss_kind in ('asym', 'mse'):
        for seed in range(5):
            t0 = time.perf_counter()
            net, ep, best = train('liu', X[trn], y[trn], X[va], y[va], seed, epochs, loss_kind)
            with torch.no_grad():
                pr = 10 ** (-net(torch.tensor(Xte, dtype=torch.float32)).numpy()[:, 0].astype(float))
            preds_all.setdefault(loss_kind, []).append(pr)
            f, U, A = rte_f[tmask], rte_U[tmask], rte_A[tmask]
            band = {}
            for lo, hi in ((5, 50), (50, 100), (100, 150)):
                b = (L >= lo) & (L < hi)
                if b.sum():
                    band['%d-%d' % (lo, hi)] = {'n': int(b.sum()), 'over_U': float(np.mean(pr[b] > U[b])), 'over_A': float(np.mean(pr[b] > A[b]))}
            out['models'].append({'loss': loss_kind, 'seed': seed, 'epochs': ep, 'best_val': best, 'train_s': time.perf_counter() - t0,
                                  'secure_prob_vs_float_label': float(np.mean(pr < f)),
                                  'overclaim_vs_U': float(np.mean(pr > U)), 'overclaim_vs_A': float(np.mean(pr > A)),
                                  'median_rel_dev': float(np.median(np.abs(pr - f) / f)),
                                  'max_overclaim_bits': float(np.max(pr - U)), 'bands': band})
            print(out['models'][-1]['loss'], seed, 'over_U %.4f over_A %.4f secure %.4f' % (out['models'][-1]['overclaim_vs_U'], out['models'][-1]['overclaim_vs_A'], out['models'][-1]['secure_prob_vs_float_label']), flush=True)
    np.savez_compressed('test_predictions.npz', mask=tmask, **{k: np.array(v) for k, v in preds_all.items()})
    json.dump(out, open(sys.argv[3], 'w'), indent=1)


if __name__ == '__main__':
    main()
