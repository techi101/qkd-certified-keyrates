# Error-enclosed key-rate certificates for DM CV-QKD

**These are mathematically proven key rates for discrete-modulated continuous-variable QKD, each checked in under half a second. Every bound comes with a witness that you can re-verify yourself.**

This is the reproducibility package for the paper *"Rounding errors break security proofs: certified key rates for continuous-variable quantum key distribution"* (N. Kumar and M. Khulbe, Netaji Subhas University of Technology). It covers asymptotic QPSK DM CV-QKD at photon-number cutoffs N_c = 8 and 12.

## Highlights

- **Every tested state certified, with no exceptions:** 120/120 under Protocol 8 and 160/160 under Protocol 10, at N_c = 8 and 12. Two independent proof routes agree on every state.
- **Fast:** the final step takes a median of **0.38 s** per state on one CPU thread. That is about 23x faster than the same enclosure without the symmetry reduction, and about 34x faster than an 80-digit mpmath recomputation, which is itself not a proof.
- **Fully enclosed net key rate:** the error-correction leakage is also evaluated with certified integration, so the reported bits per symbol carry a proof end to end.
- **Floating point measured against exact values:**
  - A reimplementation of the openQKDsecurity v2 final-step rule, applied to this protocol, returned a value above the exact value of its own expression in **51 and 60 of 120 states**.
  - A dense stress test at small depolarisation overstated by up to **0.23 nats** and reported **unsupported positive key in 7 of 120 states**.
  - The enclosed certificate never overclaims, by construction.
- **Frozen protocols:** each protocol was frozen and hashed before its run, and each runner checks its source hashes before computing.
- **Independent verifier:** it recomputes every endpoint from the released witnesses alone, without rerunning any optimisation. It refuses tampered witnesses.

## Verify a supplied certificate (no optimisation is rerun)

```bash
cd verify
python -m venv env && . env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt                 # python-flint==0.9.0, numpy, scipy
python verify.py ../data/out_p10r.json.gz --tol 1e-12      # all 160 Protocol 10R records
python verify.py ../data/out_p10r.json.gz --tol 1e-12 --n 6 # quick check, about 1 min
```

For every record, the verifier recomputes the following from the saved witness alone:
- the route-A certified lower bound, from the saved state (log-eigenvalues `w`, eigenvectors `V`, multipliers `lambda`);
- the two-sided Arb enclosure `[lo, hi]` of the exact certificate of each saved dense matrix `S` (6 values of eps);
- the enclosed error-correction leakage and the net-rate bounds.

**Two modes:**
- **Strict mode** (`--tol 0`, the default) requires every recomputed endpoint to be bit-identical to the saved one. On the original platform (Kaggle CPU, numpy 2.1.3) it passes 160/160 (`data/verify_report.json`) and, with a different numpy version, `data/verify_clean_full.json`.
- **Cross-platform mode** (`--tol 1e-12`) handles a different BLAS. The route-A lower end and each dense lower end start from a floating-point eigenvalue, so their last bits can move by about 1e-14. The recomputed values are still Arb-proven bounds. Every upper end, the leakage and the net rates must still match bit for bit. On a Windows laptop (numpy 2.3.4) it gives `VERIFIED 160/160 records; refused 0`, and the largest lower-end deviation is 4.9e-14 (`data/verify_crossplatform_tol.json`).

None of these last-bit differences changes any count reported in the paper.

### Refusal test (the verifier must reject tampered witnesses)

```bash
python refusal_test.py ../data/out_p10r.json.gz
python verify.py tampered.json --procs 3 --out refusal_result.json
```

Expected output: `VERIFIED 3/6 records; refused 3`. The three tampered records are refused: an indefinite `S`, an upper endpoint raised by 1e-6 nats, and perturbed multipliers. The three untouched records pass. The stored result is `data/verify_refusal.json`.

## Contents

| Path | What |
|---|---|
| `protocols/` | Frozen protocols 7, 8, 9, 10, 10R and 11, with their source and point hashes |
| `code/p7` | Protocol 7: the earlier frozen run (80 points, square-root factors, learned warm start) |
| `code/p8` | Protocol 8 runner: error-enclosed certificate, routes A and B (Kaggle `techie1011/qkd-v8-revision`); `enclosed_leak_p8.py` encloses the leakage |
| `code/p9` | Protocol 9: neural key-rate predictor audit (Kaggle `techie1011/qkd-v9-nn-audit`) |
| `code/trial1` | Trial 1: the exploratory dense floating-point recomputation on the Protocol 8 states, run before Protocol 10 was frozen |
| `code/p10_p10r` | Protocol 10 / 10R runners: dense floating-point evaluation vs exact enclosure, and witness export; also `lmin_witness.py` and `floor_on_witness.py` |
| `code/p11` | Protocol 11: the reimplemented openQKDsecurity v2.2.0 final-step rule vs the exact value (`tw2.py` lists deviations D1-D7) |
| `verify/` | Standalone verifier (`verify_witness.py` is frozen; `verify_tol.py` adds the cross-platform mode), refusal test, requirements |
| `data/out_p10r.json.gz` | 160 Protocol 10R records with full witnesses (720 dense matrices) |
| `data/out_p10.json` | Original Protocol 10 output, reproduced bit for bit by Protocol 10R |
| `data/out_verify8.json.gz`, `data/p8_enclosed_leak_out.json` | Protocol 8 output with saved states, and its enclosed leakage and net rates |
| `data/out_verify7.json` | Protocol 7 output and log |
| `data/trial1/` | Trial 1 outputs and summary |
| `data/p11_set1.json`, `data/p11_set2.json`, `data/p11_sanity/` | Protocol 11 on the Protocol 8 and Protocol 10R states, and its sanity gate |
| `data/verify_*.json` | Verifier runs: original platform, clean environment, cross-platform, refusal test |
| `data/lmin_witness.json`, `data/floor_on_witness.json.gz` | Inputs to the data paragraph after Proposition 2 |
| `data/audit.json` | Protocol 9 summary |

## Rerunning the experiments

Each runner for Protocols 7 to 10R checks its source hashes against `protocols/SOURCE_HASHES_*.json` and refuses to run on a mismatch.

- **Platforms:** Protocols 7, 8, 9, 10 and 10R ran on Kaggle CPU notebooks (numpy 2.1.3, scipy 1.16.3, python-flint 0.9.0, one BLAS thread, four processes). Trial 1 and Protocol 11 ran in WSL Ubuntu 24.04.
- **What depends on the platform:** floating-point outputs depend on the platform; the Arb enclosures do not. This is why the witnesses themselves are released.
- **Not released:** the 80-digit mpmath recomputation saved only its timings, which are in `out_verify8`.

**Record-keeping notes, for full transparency:**
- `FREEZE_11.json` was last written after the Protocol 11 set-1 run, when the set-2 hash was added. The source hashes inside it match the code that ran.
- `run_tw2.py` does not check its source hash.
- `run_tw2.py` exports the leakage with plain rounding rather than outward rounding. This cannot change any Protocol 11 outcome, because the smallest margin is 1.9e-4 bit.

## Scope

The guarantee is arithmetic. It proves the stated lower bound for the finite-cutoff, asymptotic optimisation problem under the simulated channel model, assuming the cutoff problem is feasible. It is not a finite-size, composable or device-level security claim. A failed floating-point certificate shows that a candidate's proof is invalid, not that the channel is insecure.

## Licence

- **Code:** MIT (see `LICENSE`).
- **Data and witnesses** (`data/`, `protocols/`): CC BY 4.0.

If you use this package, please cite the paper.
