"""Mismatched-candidate size-bias control."""

from __future__ import annotations

import numpy as np


def mismatched_partner_indices(
    n_pairs: int,
    *,
    seed: int,
    n_permutations: int = 5,
) -> np.ndarray:
    """Return shape ``(n_permutations, n_pairs)`` partner indices.

    For each permutation ``p`` and pair ``i``, ``out[p, i]`` is a partner
    index ``!= i``. Uses a fixed RNG seed; never returns self-pairs.
    """
    if n_pairs < 2:
        raise ValueError("Mismatched control requires at least 2 pairs")
    rng = np.random.default_rng(seed)
    out = np.empty((n_permutations, n_pairs), dtype=np.int64)
    for p in range(n_permutations):
        for i in range(n_pairs):
            # Sample uniformly from {0..n-1} \\ {i}
            j = int(rng.integers(0, n_pairs - 1))
            if j >= i:
                j += 1
            out[p, i] = j
    # Sanity: never self
    assert not np.any(out == np.arange(n_pairs)[None, :])
    return out
