# Protocol 11 (FROZEN 7 Nov 2026 before the full run): openQKDsecurity v2.2.0 final-step rule, float vs exact (test T-W2)

Status: frozen 7 Nov 2026 by the coordinator before the full run; sha256 of code and inputs in FREEZE_11.json. Only a 3-state smoke test has been run (Protocol 8 states
k = 0, 1, 2; see "Smoke test" below). The coordinator freezes and hashes this file before the full run.

## Question
Does the final-step rule of openQKDsecurity v2.2.0 (perturbation after the key map, eps chosen so that the smallest
eigenvalue reaches safeCut = 1e-14 lambda_max, logmsafe floor at safeCut, second perturbation of the logs,
division by 1 - eps), evaluated in binary64, give a value above the exact value of the same expression for the
same candidate on asymptotic QPSK DM CV-QKD at Nc = 8?

## Code (hash both files at freeze)
- `src/tw2.py`: `tw2_state(w, V, lam, params)`. The float side follows v2.2.0 line by line, and the docstring lists
  deviations D1-D7 (the main one: lambda_min at the given solver multipliers in place of the CVX dual). The exact
  side encloses the same expression in Arb (python-flint 0.9.0), 400 bits then 800, through the 36 x 36
  reductions in the docstring. It imports `t1_standard_float.py` and `fastcheck.py` from
  `research_07nov_qkd_p10/src` unmodified (record their sha256 at freeze).
- `src/run_tw2.py`: 4-process pool, one JSON per state. Leakage is enclosed by `enclosed_leak.leakage_bits`
  (P10 src, 128 bits, beta = 95/100).

## States
1. All 120 saved Protocol 8 states: `research_05nov_qkd_gap/data_out_verify8.json` (cohort A 80, cohort B 40).
   Note: these were already used in Trial 1 of the dense recipe, but never with this rule.
2. Later: the 120 Protocol 10R Nc = 8 witness states (file to be supplied; format as above; cohorts C10-A 80,
   C10-B 40). Record the file's sha256 at freeze.
No other states, no reruns with other settings, no tuning of eps, safeCut or multipliers.

## Per-state quantities
float_value (nats); exact_lo, exact_hi (nats, outward rounded); eps_used, eps0, safecut; leak_lo, leak_hi,
leak_mid (bits); status; prec; timings.

## Pre-declared outcomes (counts per cohort and in total, separately for each state set)
- **P1 (float above exact):** float_value > exact_hi. Report the count, the count with float_value - exact_hi > 1e-6
  nats, and the max and median of float_value - exact_hi over P1 states.
- **P2 (false key):** float net rate float_value/ln2 - leak_mid > 0, while the exact upper net rate
  exact_hi/ln2 - leak_lo < 0. Report the count and the most negative exact upper net rate among them.
- **P3 (float below exact):** float_value < exact_lo. Report the count and the max of exact_lo - float_value. This is
  conservative, not an error.
- **Status:** count of states with status != 'ok' (enclosure failures). They are reported, not excluded silently:
  the counts above are over status = 'ok' states, with the denominator stated.

## Pre-declared reading
- If P1 > 0 on either set: "a reimplementation of openQKDsecurity v2.2.0's final-step rule (deviations D1-D7)
  returned values provably above the exact value of the same expression in P1 of n states (P2 false keys)."
- If P1 = 0 on both sets: "on these states openQKDsecurity v2.2.0's rule did not overstate; the failure in Sec. 5.1
  is specific to candidate-side perturbation with a deep log floor." The paper then says so.
- Either way, no claim about openQKDsecurity's SDP dual (deviation D1) and no claim that the v2 expression with the
  second perturbation is a theorem-backed bound (D7).

## Gates before reading outcomes
- The sanity run passes: `run_tw2.py --sanity` (floor disabled, eps = 1e-6) on the first 3 states of each set gives
  float_value within 1e-9 nats of [exact_lo, exact_hi].
- Source hashes match the frozen ones. Compute is Kaggle CPU or WSL, 4 processes; record which.

## Smoke test (7 Nov 2026, WSL Ubuntu-24.04, python-flint 0.9.0, 3 processes, Protocol 8 states k = 0, 1, 2, all cohort A)
| k | eps used | safeCut | float - exact_hi (nats) | exact width | exact_s |
|---|---|---|---|---|---|
| 0 | 1.12e-12 | 7.8e-15 | +9.46e-6 | 1.0e-12 | 2.7 s |
| 1 | 8.63e-13 | 6.0e-15 | +1.46e-5 | 1.0e-12 | 2.0 s |
| 2 | 1.22e-12 | 8.5e-15 | -1.81e-5 (below exact_lo) | 1.3e-12 | 2.0 s |
Sanity (no floor, eps = 1e-6): float - [lo, hi] offsets -4.6e-10, +5.5e-13, +3.8e-10 nats (all within 1e-9).
All certified at 400 bits. Exact side about 2-2.7 s per state; total about 3.5 s per state including leakage
(one process per state, 3 in parallel). Expected full run: 120 states / 4 processes at about 4 s each, so about
2-3 min per state set.
These smoke states are part of set 1. The smoke numbers are exploratory and do not count toward the outcomes.
