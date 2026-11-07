# Error-enclosed key-rate certificates for DM CV-QKD: code, witnesses and verifier

Reproducibility package for the paper *"Floating-point key-rate certificates for discrete-modulated CV-QKD can be
invalid: sub-second error-enclosed certificates"* (asymptotic QPSK DM CV-QKD, photon-number cutoff N_c = 8 and 12).

## Verify a supplied certificate (no optimisation is rerun)

```bash
cd verify
python -m venv env && . env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt                 # python-flint==0.9.0, numpy, scipy
python verify.py ../data/out_p10r.json.gz       # all 160 Protocol 10R records, about 15 min on 4 cores
python verify.py ../data/out_p10r.json.gz --n 6 # quick check, about 1 min
```

Expected output of the full run: `VERIFIED 160/160 records; refused 0`.

For every record the verifier recomputes from the saved witness alone:
- the route-A certified lower bound, from the saved state (log-eigenvalues `w`, eigenvectors `V`, multipliers `lambda`);
- the two-sided Arb enclosure `[lo, hi]` of the exact certificate of each saved dense matrix `S` (6 values of eps);
- the enclosed error-correction leakage and the net-rate bounds.

A record passes only if every recomputed endpoint equals the saved one. The saved matrices are exact binary64 values,
and all certified arithmetic is in Arb ball arithmetic with outward rounding, so results do not depend on the BLAS or
numpy version (checked: Kaggle numpy 2.1.3 vs a clean environment with numpy 2.5.4, identical endpoints).

### Refusal test (the verifier must reject tampered witnesses)

```bash
python refusal_test.py ../data/out_p10r.json.gz
python verify.py tampered.json --procs 3 --out refusal_result.json
```

Expected: `VERIFIED 3/6 records; refused 3`. The three tampered records (an indefinite `S`, an upper endpoint raised by
1e-6 nats, perturbed multipliers) are refused; the three untouched records pass. Stored result: `data/verify_refusal.json`.

## Contents

| Path | What |
|---|---|
| `protocols/` | Frozen protocols 8, 9, 10, 10R, 11 and their source / point hashes (frozen before each run) |
| `code/p8` | Protocol 8 runner: error-enclosed certificate, routes A and B (Kaggle `techie1011/qkd-v8-revision`) |
| `code/p9` | Protocol 9: neural key-rate predictor audit (Kaggle `techie1011/qkd-v9-nn-audit`) |
| `code/p10_p10r` | Protocol 10 / 10R runners (dense floating-point evaluation vs exact enclosure; witness export), `lmin_witness.py` (certified lambda_min of every stored S), `floor_on_witness.py` (clipping-floor test) |
| `code/p11` | Protocol 11: reimplemented openQKDsecurity v2.2.0 final-step rule vs exact value (`tw2.py` lists deviations D1-D7) |
| `verify/` | Standalone verifier, refusal test, requirements |
| `data/out_p10r.json.gz` | 160 Protocol 10R records with full witnesses (720 dense matrices) |
| `data/out_p10.json` | Original Protocol 10 output (no witnesses; reproduced bit for bit by 10R) |
| `data/out_verify8.json.gz` | Protocol 8 output with saved states |
| `data/p11_set1.json`, `data/p11_set2.json` | Protocol 11 on the Protocol 8 and Protocol 10R states |
| `data/verify_report.json` | Verifier run on Kaggle (0 mismatches) |
| `data/verify_clean_full.json` | Verifier run in a clean environment (see above) |
| `data/lmin_witness.json`, `data/floor_on_witness.json.gz` | Inputs to Proposition 2's data paragraph |
| `data/audit.json` | Protocol 9 summary |

## Rerunning the experiments

Each runner checks its source hashes against `protocols/SOURCE_HASHES_*.json` and refuses to run on mismatch. Protocols
8, 9, 10 and 10R ran on private Kaggle CPU notebooks (numpy 2.1.3, scipy 1.16.3, python-flint 0.9.0, one BLAS thread,
four processes); Protocol 11 ran in WSL Ubuntu 24.04. Floating-point outputs (not the Arb enclosures) depend on the
platform; this is why the witnesses themselves are released.

## Scope

The guarantee is arithmetic and conditional: it proves the stated lower bound for the finite-cutoff, asymptotic
optimisation problem under the simulated channel model. It is not a finite-size, composable or device-level security
claim. A failed floating-point certificate shows that a candidate's proof is invalid, not that the channel is insecure.

## Licence

[AUTHORS: choose a licence before making the repository public, e.g. MIT for code and CC BY 4.0 for data.]
