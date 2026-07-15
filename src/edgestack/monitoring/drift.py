"""Feature-distribution drift via the population stability index (PSI).

PSI < 0.1: stable; 0.1-0.2: moderate shift; > 0.2: material drift — an edge
whose driving features have drifted materially is flagged.
"""

from __future__ import annotations

import numpy as np

from edgestack.exceptions import ValidationError

PSI_MATERIAL = 0.2


def population_stability_index(
    reference: np.ndarray, recent: np.ndarray, n_bins: int = 10
) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = np.asarray(recent, dtype=float)
    cur = cur[~np.isnan(cur)]
    if len(ref) < 20 or len(cur) < 10:
        raise ValidationError("PSI needs at least 20 reference and 10 recent points")

    # Bin edges from the REFERENCE distribution's quantiles.
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:  # nearly constant feature: no measurable drift
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_frac = np.clip(ref_counts / len(ref), eps, None)
    cur_frac = np.clip(cur_counts / len(cur), eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))
