# Protocol 10R: witness-export reproduction of Protocol 10 (frozen 7 Nov 2026, before the run)

Purpose: Protocol 10 (sha256 17d27d0f1672..., Kaggle techie1011/qkd-v10-dense-recipe v1) did not save its certificate
witnesses, and it exported the leakage endpoints with plain float() conversion, which does not round outward
(external audit, 7 Nov). Protocol 10R repeats the IDENTICAL computation (same points, seeds, solver, eps grid, checker,
precision, source files byte-identical to Protocol 10) and changes only what is saved and how the final rates are formed.
The original Protocol 10 outputs are kept unchanged.

## Changes relative to Protocol 10 (no change to any computed certificate)
1. Leakage endpoints exported outward with fastcheck.lo()/up() (nextafter of the Arb endpoint).
2. Net rates formed in Arb from the leakage ball (128 bits):
   - route A net-rate lower bound = lo(A/ln2 - leak)
   - dense candidate net-rate lower bound = lo(lo_c/ln2 - leak); upper bound = up(hi_c/ln2 - leak)
   - float claim (what a float pipeline reports) = float_std/ln2 - mid(leak) in binary64
3. Witnesses saved: solver state (w, V, lifted lambda) and, for every eps, the exact binary64 matrix S that was enclosed.
4. Standalone verifier (verify_witness.py) re-encloses every saved S and every route-A state from the saved witness only,
   and must reproduce the saved endpoints exactly.

## Gates
- R1 reproduction: route-A bounds, float certificates and dense (lo, hi, float_std) equal Protocol 10 bit for bit
  in all 160 / 720 records. Any mismatch is reported, not hidden; on mismatch the 10R values are the ones with witnesses.
- R2 verifier: verify_witness.py reproduces every saved endpoint from the witness file.
## Outcomes (same definitions as Protocol 10, recomputed with the corrected rates)
P1 float_std > hi_c; P2 float claim > 0 and candidate upper net rate < 0; P3 proven-positive counts; P4 Nc=12.
