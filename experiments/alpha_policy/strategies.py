"""
Strategy baskets for the alpha-policy experiment.

Each strategy returns up to K product indices for a target country. When a
feasible mask is provided, all selected products satisfy beta_C[target, i] > 0.

The *_rca strategies and the off-grid threshold rank by revealed comparative
advantage (RCA), passed in as `rca_target`. RCA is the Balassa index computed
on the same export matrix the rest of the pipeline uses (see
calibration/data_extraction_all_years.compute_rca_from_exports), so RCA > 1
means the country is specialised in that product. If `rca_target` is not
provided, the functions fall back to ranking by the allocation share
`alpha_target` (legacy behaviour, used only by unit tests).

- top_rca:                 K largest rca[target, :] (strongest comparative advantage)
- bottom_rca:              K smallest rca[target, :] (anti-strategy / control)
- proximity:               K largest model-consistent density values, excluding top_rca.
                           density[i] = (sum_{i'} phi[i, i'] * alpha[i']) / mean_{i'} phi[i, i']
                           (weighted by the country's current export footprint alpha)
- random:                  K uniformly-random products (single seeded draw)
- high_complexity_offgrid: K highest-PCI products the country is NOT specialised in
                           (rca[target, i] < 1; or, in the alpha fallback, below the
                           median allocation share). If `pci` is None, falls back to a
                           clearly-labeled proxy (sum of phi-row, a within-graph
                           centrality) and emits a warning.
"""
import warnings
import numpy as np


STRATEGY_NAMES = (
    "top_rca",
    "bottom_rca",
    "proximity",
    "random",
    "high_complexity_offgrid",
)


def _density(alpha_target: np.ndarray, phi_space: np.ndarray) -> np.ndarray:
    """Model-consistent density: (phi @ alpha) / row-mean(phi)."""
    if phi_space.shape != (alpha_target.shape[0], alpha_target.shape[0]):
        raise ValueError(
            f"phi_space shape {phi_space.shape} incompatible with "
            f"alpha length {alpha_target.shape[0]}"
        )
    row_mean = phi_space.mean(axis=1)
    row_mean = np.where(row_mean > 0, row_mean, 1.0)
    return (phi_space @ alpha_target) / row_mean


def build_strategies(
    alpha_target: np.ndarray,
    phi_space: np.ndarray,
    K: int = 10,
    seed: int = 0,
    pci: np.ndarray = None,
    feasible_mask: np.ndarray = None,
    rca_target: np.ndarray = None,
) -> dict:
    """Build 5 strategy baskets. Returns {name: int-indices array}.

    `rca_target` (1-D, length SP) is the country's revealed comparative
    advantage; it drives top_rca, bottom_rca, and the off-grid threshold. When
    omitted, those strategies fall back to the allocation share `alpha_target`
    (legacy behaviour).

    If feasible_mask is provided, all baskets are restricted to feasible
    products. In that case baskets may be shorter than K when fewer than K
    products are feasible for the target country.
    """
    alpha_target = np.asarray(alpha_target, dtype=float)
    if alpha_target.ndim != 1:
        raise ValueError(f"alpha_target must be 1-D, got shape {alpha_target.shape}")
    phi_space = np.asarray(phi_space, dtype=float)
    SP = alpha_target.shape[0]
    if rca_target is None:
        rca_score = alpha_target
        rca_provided = False
    else:
        rca_score = np.asarray(rca_target, dtype=float)
        if rca_score.shape != (SP,):
            raise ValueError(
                f"rca_target shape {rca_score.shape} != ({SP},)"
            )
        rca_provided = True
    if int(K) < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K > SP:
        raise ValueError(f"K={K} exceeds SP={SP}")
    if feasible_mask is None:
        feasible = np.ones(SP, dtype=bool)
    else:
        feasible = np.asarray(feasible_mask, dtype=bool)
        if feasible.shape != (SP,):
            raise ValueError(f"feasible_mask shape {feasible.shape} != ({SP},)")
    feasible_idx = np.where(feasible)[0]
    if feasible_idx.size == 0:
        raise ValueError("feasible_mask has no feasible products")
    K_eff = min(int(K), int(feasible_idx.size))

    def _ranked_pick(score, descending=True, mask=None):
        candidate_mask = feasible.copy()
        if mask is not None:
            candidate_mask &= np.asarray(mask, dtype=bool)
        candidates = np.where(candidate_mask)[0]
        if candidates.size == 0:
            return np.empty(0, dtype=int)
        order = candidates[np.argsort(score[candidates])]
        if descending:
            order = order[::-1]
        return order[:K_eff].astype(int)

    out = {}

    # top_rca
    out["top_rca"] = _ranked_pick(rca_score, descending=True)

    # bottom_rca
    out["bottom_rca"] = _ranked_pick(rca_score, descending=False)

    # proximity (exclude top_rca)
    dens = _density(alpha_target, phi_space)
    excl = set(out["top_rca"].tolist())
    order = feasible_idx[np.argsort(dens[feasible_idx])][::-1]
    pick = [int(i) for i in order if int(i) not in excl][:K_eff]
    if len(pick) < K_eff:
        # If the feasible set is small, fill from top-density feasible products.
        for i in order:
            ii = int(i)
            if ii not in pick:
                pick.append(ii)
            if len(pick) >= K_eff:
                break
    out["proximity"] = np.asarray(pick, dtype=int)

    # random
    rng = np.random.default_rng(seed)
    out["random"] = np.sort(
        rng.choice(feasible_idx, size=K_eff, replace=False)
    ).astype(int)

    # high_complexity_offgrid: high PCI among products the country is NOT
    # specialised in. With RCA available, "not specialised" = RCA < 1 (the
    # standard Balassa threshold); otherwise fall back to below-median share.
    if rca_provided:
        offgrid_mask = feasible & (rca_score < 1.0)
    else:
        median_alpha = np.median(alpha_target[feasible])
        offgrid_mask = feasible & (alpha_target < median_alpha)
    if pci is None:
        warnings.warn(
            "PCI not provided to build_strategies; using phi-row sum as a proxy. "
            "Strategy 'high_complexity_offgrid' is a labeled approximation.",
            stacklevel=2,
        )
        complexity = phi_space.sum(axis=1)
    else:
        complexity = np.asarray(pci, dtype=float)
        if complexity.shape[0] != SP:
            raise ValueError(f"pci length {complexity.shape[0]} != SP {SP}")
    idx = _ranked_pick(complexity, descending=True, mask=offgrid_mask)
    if idx.size < K_eff:
        # Not enough feasible off-grid products; fall back to top-complexity
        # feasible products.
        idx = _ranked_pick(complexity, descending=True)
    out["high_complexity_offgrid"] = idx.astype(int)

    return out
