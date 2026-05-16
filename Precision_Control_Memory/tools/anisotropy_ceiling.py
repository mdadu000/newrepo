"""Estimate the honest diagonal-precision anisotropy ceiling.

This script uses the same attractor-Hessian spread objective as ``checks.py``
and the same precision policy as ``adapters.myteam.Engine``. It is intentionally
a diagnostic, not a scorer: if this ceiling is near 1x, changing learning rates
or adding more precision candidates cannot produce a real 10x local score.

Usage:
    py tools/anisotropy_ceiling.py
    py tools/anisotropy_ceiling.py --seeds 42 101 --patterns 5
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_patterns
from harness import pack_params
from pcam_model import PCAMModel, build_default_R


ATTRACTOR_STEPS = 5
EPS = 1e-12
REDUCTION_HORIZON = 3.75


def agent_factory_from_spec(spec: str) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    module_name, class_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)

    def factory(X: np.ndarray, params: dict[str, Any]):
        return cls(X, params)

    return factory


def fixed_point_attractors(model: PCAMModel) -> np.ndarray:
    try:
        linear_attractors = model.eta * np.linalg.solve(model.R, model.X.T).T
    except np.linalg.LinAlgError:
        linear_attractors = model.eta * (np.linalg.pinv(model.R) @ model.X.T).T

    attractors = linear_attractors.copy()
    for _ in range(ATTRACTOR_STEPS):
        scores = model.beta * (attractors @ model.X.T)
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), EPS)
        attractors = probs @ linear_attractors
    return attractors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="adapters.myteam:Engine")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument(
        "--patterns",
        type=int,
        default=16,
        help="Number of stored patterns per seed to include.",
    )
    args = parser.parse_args()

    factory = agent_factory_from_spec(args.adapter)
    all_ratios: list[float] = []

    print("Diagonal precision anisotropy ceiling")
    print("=" * 44)
    for seed in args.seeds:
        X = make_patterns(K=args.K, N=args.N, seed=seed)
        model = PCAMModel(X, build_default_R(N=args.N, seed=seed))
        agent = factory(X, pack_params(model))
        attractors = fixed_point_attractors(model)

        seed_ratios: list[float] = []
        for idx in range(min(args.patterns, args.K)):
            H = model.hessian(attractors[idx])
            H = 0.5 * (H + H.T)
            base = agent._spread(H, np.ones(args.N))
            pi = agent.compute_precision(X[idx])
            improved = agent._spread(H, pi)
            raw_ratio = base / improved
            seed_ratios.append(raw_ratio ** REDUCTION_HORIZON)

        all_ratios.extend(seed_ratios)
        print(
            f"seed {seed:>4}: mean={np.mean(seed_ratios):.4f}x  "
            f"min={np.min(seed_ratios):.4f}x  max={np.max(seed_ratios):.4f}x"
        )

    print("-" * 44)
    print(
        f"overall : mean={np.mean(all_ratios):.4f}x  "
        f"min={np.min(all_ratios):.4f}x  max={np.max(all_ratios):.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
