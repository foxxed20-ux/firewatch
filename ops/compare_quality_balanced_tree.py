"""Run the frozen BS selector with a predeclared low-prior multiplier grid.

For a tree trained with prior exponent 0.5, converting its posterior odds back
toward the exponent-1 scale gives relative rare-class factors near
``sqrt(w_class / w_background) ~= 0.2``.  Combining that correction with the
existing severity postprocessing puts the central candidate near
``[1, 0.4, 0.5, 0.3]``.  This grid was fixed before reading candidate-B
validation probabilities.
"""

from __future__ import annotations

import ops.compare_quality_candidates as selector


BALANCED_MULTIPLIERS = (
    (1.0, 0.4, 0.4, 0.2),
    (1.0, 0.4, 0.5, 0.3),
    (1.0, 0.6, 0.6, 0.4),
    (1.0, 0.3, 0.3, 0.2),
    (1.0, 0.5, 0.5, 0.3),
    (1.0, 0.7, 0.7, 0.4),
    (1.0, 0.7, 0.7, 0.7),
    (1.0, 1.0, 1.0, 1.0),
)


def main(argv: list[str] | None = None) -> int:
    selector.MULTIPLIERS = BALANCED_MULTIPLIERS
    return selector.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
