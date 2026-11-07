# Protocol 9 (frozen before any outcome): auditing published-style neural key-rate predictors

**Question:** how often does a neural key-rate predictor, built and trained as in Liu et al., Opt. Express 30, 15024 (2022), claim more key than can be certified? Liu et al. is the AutoML follow-up of Zhou et al., Sci. Rep. 12, 8879 (2022), for the same QPSK heterodyne protocol (Lin et al. 2019).

## Data (gen_data.py)
**Points:** uniform in L [5, 150] km, xi [0.0045, 0.0405] (the published xi range), alpha [0.62, 0.72] (the published alpha range).
- Train: 40,000 points from SeedSequence([20261103, 1, 0]).
- Test: 8,000 points from SeedSequence([20261103, 2, 0]).

**Labels:** cold-start solver (same as Protocols 6-8) to a 1e-6 final gap.
- float_cert is the numerical-method label.
- U is the approximate primal objective (approximate upper reference for f*).
- delta_EC uses beta = 0.95.
- Test points also get the certified bound A (FastChecker2, route A).

**Inputs:** the 29 published variables: 16 conditional Bob moments, 12 off-diagonal Alice terms and xi. Standardised with training statistics.

## Models (train_eval.py)
- **Architecture:** Liu et al. Table III. Adam, lr 1e-3, batch 64, at most 200 epochs, early stopping on a 10% validation split (patience 20).
- **Labels:** -log10(rate). As published, only points with a positive float key rate are used.
- **Losses:**
  - (a) the published asymmetric loss with gamma = 0.0539 and eps = 0.8727, read from Table III; assignment of the two numbers to gamma and eps assumed from the search ranges (0.05, 0.2) and (0.8, 0.99);
  - (b) MSE.
- 5 seeds each.

## Outcomes (test points with positive float rate and a certified A)
- **Primary:** the overclaim rate, the fraction of predictions above the U-based rate. This is a claim of more key than the optimum allows (up to U's approximate feasibility). It is reported per model, as the mean over the 5 asymmetric-loss seeds, and by distance band.
- **Secondary:**
  - predictions above the certified A-based rate;
  - the published "secure probability" (prediction below the float label);
  - median relative deviation;
  - maximum overclaim in bits.
- **Reference:** the certified pipeline (solver + route A) overclaims in 0 cases by construction. Its cost is reported from Protocol 8.

**No gate.** This is a descriptive audit. Whatever the rates are, they are reported, including if they are ~0.

**Deviation from the published work:**
- smaller data (40k vs 769k);
- our solver and certificate instead of theirs;
- no TPE search (their selected architecture is reused).

These are stated in the paper.
