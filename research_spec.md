# BTC Valuation Research — Frozen Research Specification

## Research question

Do candidate economic anchors for Bitcoin produce valuation relationships
that are statistically defensible, stable through time and robust out of sample?

Secondary questions:

1. Does a statistically defensible long-run relationship exist?
2. Does the relationship remain stable and useful out of sample?
3. Does the valuation gap contain information about future Bitcoin returns?

No causal interpretation will be made without a specific identification strategy.

---

## Core notation

Observed Bitcoin market price:

P_t

Model-implied value for model m:

V_hat_t^(m) = f_m(X_t ; theta_hat)

Valuation gap:

g_t^(m) = log(V_hat_t^(m) / P_t)

Convention:

g_t > 0 => model-implied value exceeds market price.

g_t < 0 => model-implied value is below market price.

This convention is frozen.

---

## Frequency

Primary frequency: monthly.

Requested sample start: 2017-08-17.

Each model may use its maximum valid historical sample.

Direct model comparisons must additionally be performed on a common sample.

No silent interpolation is permitted.

Every exclusion must be logged.

---

## Candidate valuation families

### 1. Scarcity

SF_t = Stock_t / Flow_t

log(P_t) = alpha + beta log(SF_t) + epsilon_t

The model is not considered valid merely because of high in-sample R².

### 2. Adoption

log(P_t) = alpha + beta log(A_t) + epsilon_t

Primary adoption proxy must be selected before inspection of model results.

Alternative proxies are robustness checks.

Association must not be interpreted as causality.

### 3. Network value

MarketCap_t = k N_t^beta

log(MarketCap_t) = alpha + beta log(N_t) + epsilon_t

Specific hypothesis:

H0: beta = 2

The existence of a network relationship and literal Metcalfe's law are
treated as separate hypotheses.

### 4. Production cost

C_t = (Energy_t + Hardware_t + Operating_t) / BTCProduced_t

Production-cost analysis must explicitly account for possible reverse
direction:

Price -> Hashrate -> Difficulty -> Cost

Lead-lag analysis, VAR and Granger tests may be used.

Granger causality must not be described as structural causality.

### 5. Gold scenario

V_BTC(theta) = theta * MarketValue_Gold / BTC_Supply

Frozen theta scenarios:

5%, 10%, 25%, 50%, 100%.

This is a scenario analysis rather than an independent intrinsic-value model.

---

## Time-series diagnostics

For each relevant series evaluate:

x_t
log(x_t)
Delta log(x_t)

Diagnostics:

- level plot
- log plot
- first differences
- ACF
- PACF where relevant
- ADF
- KPSS
- outlier diagnostics
- variance changes

Integration order must be assessed before level regressions are interpreted.

---

## Cointegration

When both variables are compatible with I(1):

log(P_t) = alpha + beta log(X_t) + epsilon_t

Test whether:

epsilon_t ~ I(0)

Primary method: Engle-Granger.

A high R² without stationary residuals is not evidence of a valid long-run
valuation relationship.

---

## Out-of-sample protocol

Strict expanding-window estimation.

At time t:

theta_hat_(t-1) is estimated only from information available through t-1.

Then:

V_hat_t = f(X_t ; theta_hat_(t-1))

P_t must never be used to estimate parameters used to construct its own
model-implied value.

---

## Baselines

Every empirical model must be compared against simple baselines.

Baseline 1:

V_hat_t = P_(t-1)

Baseline 2:

rolling mean of historical log prices.

Complexity is useful only if it provides incremental empirical information.

---

## Error metrics

e_t = log(P_t) - log(V_hat_t)

Primary metrics:

MAE
RMSE
Bias
Median Absolute Error

All must be reported.

---

## Statistical inference

HAC / Newey-West standard errors must be used where serial dependence or
overlapping returns makes classical OLS inference inappropriate.

For horizon h:

HAC lag >= h - 1.

Block bootstrap must be used as an additional robustness procedure.

Bootstrap block-length selection must be determined before inspecting model
results.

Default bootstrap replications: 5,000.

---

## Parameter stability

Full-sample coefficients are insufficient.

Required:

- rolling coefficient estimates
- confidence intervals
- predefined-break Chow tests where justified
- multiple structural-break detection

A model with strong full-sample fit but materially unstable coefficients is
not considered structurally robust.

---

## Predictive regressions

Only after construction of out-of-sample valuation gaps:

R_(t,t+h) = alpha_h + beta_h g_t + epsilon_(t+h)

Frozen horizons:

3 months
6 months
12 months
24 months

Primary null:

H0: beta_h = 0

Overlapping-return inference must use HAC and bootstrap robustness.

A null predictive result is a valid research result.

---

## Multiple testing

Primary hypotheses and exploratory analyses must be separated.

Primary families of tests use Holm correction.

Unadjusted p-values may be reported, but primary conclusions must respect
multiplicity-adjusted inference.

---

## Sensitivity analysis

Every valuation family must expose its major modelling assumptions.

Scarcity:
flow definition.

Adoption:
proxy choice.

Network:
network proxy and beta.

Production:
energy cost, hardware efficiency and operating assumptions.

Gold:
theta.

The research must distinguish information supplied by data from information
introduced through modelling assumptions.

---

## Falsification criteria

A model is materially weakened by combinations of:

- absence of the required long-run relationship
- parameter instability
- dependence on one sub-period
- poor out-of-sample performance
- failure to improve on naïve baselines
- fragility to reasonable modelling changes
- disappearance after multiple-testing correction
- excessive dependence on externally imposed assumptions

Negative findings must not be removed or reframed after inspection.

---

## Research integrity rules

No future information.

No silent interpolation.

No silent deletion.

No post-result model selection presented as confirmatory.

No causal language without causal identification.

No model called "fundamental value" merely because it fits historical price.

All datasets, transformations, exclusions and model versions must be logged.