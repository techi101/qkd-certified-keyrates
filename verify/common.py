"""Shared helpers for 5 Nov gap trials (copies of frozen code are in this folder; originals untouched)."""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def load_states():
    d = json.loads((ROOT / 'data_out_verify8.json').read_text())
    out = []
    for r in d['res']:
        s = r['state']
        V = np.array(s['V_re']) + 1j * np.array(s['V_im'])
        out.append(dict(rec=r, st={'w': np.array(s['w']), 'V': V, 'lambda': np.array(s['lambda_source'])}))
    return out
