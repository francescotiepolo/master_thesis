"""
Deterministic alpha-injection for the alpha-policy experiment.

The injected counterfactual alpha row is a function of:
  - the observed alpha row at that year,
  - a basket of K product indices,
  - a strength parameter w in [0, 1].

`w=0` reproduces observed alpha exactly (baseline). `w=1` is the maximal
policy intervention. The output is non-negative and sums to 1.
"""
import numpy as np


def _normalise(row):
    row = np.clip(np.asarray(row, dtype=float), 0.0, None)
    s = row.sum()
    if s > 0:
        return row / s
    if row.size == 0:
        return row
    return np.full(row.size, 1.0 / row.size)


def inject(alpha_obs_row, basket, strength):
    """
    Transfer a fraction w of the off-basket mass into the basket, distributing
    it proportionally to each basket product's existing alpha share.

      alpha_cf[i] = (1-w) * alpha_obs[i]                              for i not in basket
      alpha_cf[i] = alpha_obs[i] + w * off_mass * a[i] / basket_mass  for i in basket

    Falls back to uniform within the basket if its current alpha mass is 0.
    """
    w = float(strength)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    a = np.asarray(alpha_obs_row, dtype=float).copy()
    if a.ndim != 1:
        raise ValueError(f"alpha_obs_row must be 1-D, got shape {a.shape}")
    basket = np.unique(np.asarray(basket, dtype=int))
    if basket.size == 0:
        return _normalise(a)
    if basket[0] < 0 or basket[-1] >= a.size:
        raise IndexError(
            f"basket indices must be in [0, {a.size - 1}], got "
            f"[{basket[0]}, {basket[-1]}]"
        )
    mask = np.zeros_like(a, dtype=bool)
    mask[basket] = True
    off_mass = a[~mask].sum()
    basket_mass = a[basket].sum()
    a[~mask] *= (1.0 - w)
    if basket_mass > 0:
        weights = a[basket] / basket_mass
    else:
        weights = np.full(basket.size, 1.0 / basket.size)
    a[basket] = a[basket] + w * off_mass * weights
    return _normalise(a)
