"""Same-label mismatched-candidate partners (never self)."""

from __future__ import annotations

import numpy as np

from app.services.gap_sensitivity.controls import mismatched_partner_indices


def mismatched_partner_indices_by_label(
    labels: list[str],
    *,
    seed: int,
    n_permutations: int = 5,
) -> np.ndarray:
    """Shape ``(n_permutations, n_pairs)``; partners share the label and are never self."""
    n = len(labels)
    if n < 2:
        raise ValueError("Mismatched control requires at least 2 pairs")
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(str(lab), []).append(i)
    out = np.full((n_permutations, n), -1, dtype=np.int64)
    for lab, idxs in groups.items():
        if len(idxs) < 2:
            raise ValueError(
                f"Same-label mismatched control needs ≥2 pairs for {lab!r}, got {len(idxs)}"
            )
        local = mismatched_partner_indices(
            len(idxs), seed=seed, n_permutations=n_permutations
        )
        idx_arr = np.asarray(idxs, dtype=np.int64)
        for p in range(n_permutations):
            out[p, idx_arr] = idx_arr[local[p]]
    if np.any(out < 0):
        raise RuntimeError("Unassigned mismatched partner")
    if np.any(out == np.arange(n)[None, :]):
        raise RuntimeError("Self-pair in mismatched control")
    for p in range(n_permutations):
        for i in range(n):
            if labels[int(out[p, i])] != labels[i]:
                raise RuntimeError("Mismatched partner has a different fit label")
    return out
