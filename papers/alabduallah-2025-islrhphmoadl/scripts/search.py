"""Literal printed CS-IGWO equations plus explicitly approved interpretations.

No intermediate candidate evaluation is performed. Equation 12's inferred
greedy repair is used only when both exact intermediate positions have cached
fitness under the same fixed training protocol; otherwise use its literal output.
"""
import numpy as np


def decode(position):
    position = np.clip(np.asarray(position, dtype=np.float64), 0, 1)
    head_lr = float(10 ** (-3 + position[0]))
    # Only classifier LR is searched. Final ViT LR stays fixed at 0.001.
    return {"head_lr": head_lr, "backbone_multiplier": 0.001 / head_lr}


def key(position):
    return tuple(float(x) for x in np.clip(position, 0, 1))


def initialize(rng):
    return np.vstack(([1.0], rng.uniform(0, 1, (4, 1))))


def propose(population, leaders, errors, t, rng, cached_scores):
    """Generate a whole population before any evaluation or leader refresh."""
    progress = t / 3
    a = 2 - 2 * progress ** 2
    gamma1, gamma2 = progress ** 2, 1 - progress ** 2
    weights = errors / errors.sum() if errors.sum() else np.full(3, 1 / 3)
    probability = 1 - 1.01 * progress ** 3  # Eq.15, deliberately NOT clipped.
    proposed, trace = [], []
    for position in population:
        r1, r2 = rng.uniform(size=(3, len(position))), rng.uniform(size=(3, len(position)))
        A, C = 2 * a * r1 - a, 2 * r2
        X = leaders - A * np.abs(C * leaders - position)  # Original GWO definitions.
        weighted = (weights[:, None] * X).sum(axis=0) / 3  # Eq.7 literally.
        mutated = (X * (1 + gamma1 * rng.normal(size=(3, len(position)))
                        + gamma2 * rng.standard_cauchy(size=(3, len(position))))).mean(axis=0)
        start = mutated
        repaired = key(mutated) in cached_scores and key(weighted) in cached_scores
        if repaired and cached_scores[key(mutated)] >= cached_scores[key(weighted)]:
            start = weighted
        branch_draw, movement = rng.uniform(), rng.uniform()
        destination = (X[0] + X[1]) / 2 if branch_draw < probability else X[0]
        raw = start + movement * (destination - start)  # fl=1, Eqs.13-14.
        if not np.isfinite(raw).all():
            raise FloatingPointError("Nonfinite search proposal; do not silently repair")
        proposed.append(np.clip(raw, 0, 1))
        trace.append({"weighted": weighted.tolist(), "mutated": mutated.tolist(),
                      "equation12_repaired_without_evaluation": repaired,
                      "crow_branch": 13 if branch_draw < probability else 14,
                      "raw_position": raw.tolist(), "probability": probability})
    return np.asarray(proposed), trace


def leaders_from_history(history):
    successful = [h for h in history if h.get("misclassified") is not None]
    ranked = sorted(successful, key=lambda h: (h["misclassified"], h["evaluation"]))[:3]
    if len(ranked) < 3:
        raise RuntimeError("Fewer than three valid leaders; cannot run search")
    return np.array([r["position"] for r in ranked]), np.array([r["misclassified"] / 4060 for r in ranked])
