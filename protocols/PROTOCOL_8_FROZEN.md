# Protocol 8 (frozen before any outcome on its cohorts): revision run after independent review

Written 2 Nov 2026 in response to REVIEW_1.md. Development used only dev seeds [20261102,7,*]. Protocol 7 test states are not reused for tuning.

## Changes from Protocol 7 (all developed on development points)
1. **Route A and route B factor R_0 by a ball Cholesky factor C (C C^dag = R_0)** instead of R_0^{1/2}.
   - Objective and gradient are unchanged: S(C^dag rho C) = S(rho^1/2 R rho^1/2).
   - Route B therefore uses no eigen-solver in any proof step.
2. **Directed rounding:** reported floats are nextafter-rounded outward from the ball ends.
3. **Every certified state is saved** (w, V, source-coordinate lambda). Both routes run on exactly the same saved states.

## Cohorts (cold-start solver only; Protocol 6/7 solver and stop rule)
- **A, standard:** 80 points from SeedSequence([20261102,10,0]), range [5,60] km x [0.005,0.04] x [0.5,0.9].
- **B, low-rate:** 40 points from SeedSequence([20261102,10,1]), range [60,100] km x [0.01,0.05] x [0.5,0.9].
- Both cohorts must be disjoint by hash from KNOWN_POINT_HASHES.json, which now also contains the Protocol 7 cohort and all development points.

## Per state
- Route A (FastChecker2), timed 3 times; take the median.
- Route B (LoewnerCrossCheck, c = 1e-9, 800 bits), timed once.
- Float certificate time: gradient plus eigvalsh from the final state, median of 3.
- Leakage delta_EC (beta = 0.95, keyrate.py, the same as qkd.p_ab). Key rate in bits = certified/ln 2 - delta_EC.
- One-time setup times are recorded: model, FastChecker2 at 1600 bits, LoewnerCrossCheck.

**Ablation on cohort A points 0-9:** FastChecker v1 (no symmetry; four eig-based logs; R^{1/2}) at 400 bits, timed once.

**Old mpmath diagnostic on cohort A points 0-5:** reported, timed once.

## Primary gates (all must pass)
1. Route A certifies all 120 states.
2. Route B certifies all 120 states, with 0 <= (A - B) <= 1e-9 + 1e-12 nats in every state.
3. |A - float certificate| <= 1e-6 nats in every state.
4. Median route-A time <= 1 s.

## Secondary (no gate)
- Time tails; float-certificate time; ablation ratio; setup times.
- Key rates and |certified - float| / |key rate| by cohort.
- Number of states where float exceeds the certified upper end.

If a gate fails, report it as is.
