# Protocol 10 (FROZEN 7 Nov 2026, before any run): confirm Trial 1 on fresh points + larger cutoff
Status: written 5 Nov 2026 after Trial 1 (exploratory, on the Protocol 8 states). Frozen 7 Nov 2026 unchanged in substance; runner = run_p10.py (source hashes in SOURCE_HASHES_10.json). Any deviation is logged in RESULTS_10.md.
## Cohorts
- C10-A: 80 fresh points, SeedSequence([20261105,10,0]), L in [5,60] km, xi in [0.005,0.04], alpha in [0.5,0.9], Nc=8.
- C10-B: 40 fresh points, SeedSequence([20261105,10,1]), L in [60,100], xi in [0.01,0.05], Nc=8.
- C10-N12: 40 fresh points, SeedSequence([20261105,10,2]), cohort-A ranges, Nc=12.
- All must be hash-disjoint from KNOWN_POINT_HASHES.json (Protocols 7-9 + dev) and from the Protocol 8 points.
## Per state
1. Cold-start solver (optimized_solver.solve_optimized), route A (FastChecker2), float certificate (log-domain).
2. Dense-matrix float recipe at eps in {1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8}: S = Model.regular(sigma_dense, eps); float value Model.cert(Model.grad(S), lam, gam);
   two-sided Arb enclosure [lo, hi] of the exact certificate of the same S (t1_standard_float.exact_bound).
3. Enclosed leakage (enclosed_leak.leakage_bits, 128 bits).
## Primary outcomes (pre-declared, reported as is)
- P1: number of states where float value > hi (proven overstatement), per eps.
- P2: number of states where the float net rate > 0 while the exact net rate of the same candidate (hi) <= 0, per eps.
- P3: positive certified net rates: route A vs dense float at each eps.
- P4 (N12): route A certifies all states; median route-A time.
## Expected cost
Trial 1 cost ~2.2 s per (state, eps) at Nc=8 on one WSL core: 120 x 6 x 2.2 s ~ 26 min single thread, plus Nc=12 (see Trial 2 timing). Kaggle CPU, 4 processes.
