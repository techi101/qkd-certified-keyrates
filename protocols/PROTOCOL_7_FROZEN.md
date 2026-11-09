# Protocol 7 (frozen before any test outcome): fast ball-arithmetic certificate

Written 2 Nov 2026, after development on 21 dev points only (FINDINGS_02NOV_1.md and the smoke-run note below). Earlier test cohorts are not reused.

**Deviation, disclosed:** QUANTUM_ADDITIONS_1.md proposed a 100 ms go/no-go for the learned-start *speed* claim. The development median was 0.255 s, so that speed claim is not expected to survive. This protocol instead tests the certificate itself. The gates below were chosen after seeing development data only, never test data.

## Cohort
- 80 points from `np.random.default_rng(np.random.SeedSequence([20261102,9,0])).random((80,3))*[55,.035,.4]+[5,.005,.5]`, giving (L km, xi, alpha).
- Must be disjoint (by SHA256 of float64 bytes) from KNOWN_POINT_HASHES.json, which covers the v6 known set, the v6 test cohort, all 21 development points and the 2 smoke-run points described below.
- QPSK, Nc=8, clean inputs.

## Methods
- Each point: 3 repetitions. In each, R6_cold and R6_ridge run in an order randomised by SeedSequence([20261102,9,2]). Both use the frozen ridge model e7bd0362..., the same solver as Protocol 6 and the same stop rule (final 1e-6 outer gap).
- **FastChecker2** (src/fastcheck.py, python-flint==0.9.0, 400 bits, whole-pipeline precision ladder 400/800/1600 bits as coded; fixed operators enclosed at 1600 bits; states with smallest log-eigenvalue below -140 start at 800 bits) runs on the repetition-0 final state of each method at every point: 160 states.
  - Each check is timed 3 times; the per-state time is the median of the 3.
  - The certified value is taken from the first call.
- The old mpmath Checker (verify_v4) runs on points 0-5, both methods, repetition 0: 12 states, timed once.
- One BLAS thread; private CPU Kaggle kernel. Internet is used only to pip-install the pinned python-flint before any computation.

## Primary gates (all must pass)
1. **Certification:** all 160 states return a certified lower bound, with no exception after the fallback ladder.
2. **Agreement:** for every state, |certified lower bound - float certificate| <= 1e-6 nats.
3. **Speed:**
   - the median per-state FastChecker2 time is <= 1.0 s;
   - on the 12 paired states, the median of (fast time / old time) is <= 0.1.

## Secondary (reported, no gate)
- Tail times: 90th percentile and maximum. Fallback usage counts.
- Old vs fast value differences on the 12 paired states.
- **Total time.** For each point, the solver time to final stop (median of 3 repetitions) plus the fast-check time of that method's state.
  - Report the median paired learned/cold total ratio, with a 10,000-replicate point bootstrap 95% interval (SeedSequence([20261102,9,3])).
  - Report the same ratio for solver time alone.
- Numerical gates as in Protocol 6 are logged: source residual <= 1e-7, trace error <= 1e-10, inner convergence.

## Interpretation
- **If the gates pass:** "On fresh simulated QPSK cutoff-model points, a ball-arithmetic certificate gave a certified lower bound on the cutoff-model key-rate objective within 1e-6 nats of the float value, with median cost X s on Kaggle CPU."
- **Not claimed:**
  - the photon cutoff;
  - an upper reference;
  - finite-size or physical security;
  - independence from Arb's experimental eigen routines.
- **If any gate fails,** report the failure as is; no re-run with changed settings counts as confirmation.

## Smoke-run incident (disclosed)
A local 2-point smoke run used the first two points of the originally planned seed [20261102,8,0]. One of them (L=5.8 km) failed to certify. This exposed a precision limit: the fixed operators had been enclosed at only 400 bits.
- The fix was developed on 6 new short-distance development points from dev seed [20261102,7,5]. All 18 development points then certified: median 0.266 s, maximum 0.89 s on the laptop.
- Because a test outcome had been seen, the whole cohort was replaced by seed [20261102,9,0]. The 2 smoke points were added to the known-point hashes.
