"""AVSPO reward-statistic augmentation before the repository HARPO adapters.

The source AVSPO paper assumes binary rewards. Reward B is composite, so an
all-wrong homogeneous group uses anchors *above its observed score* and an
all-correct homogeneous group uses anchors below it. This preserves the
paper's sign while leaving non-collapsed groups and all HARPO scales intact.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class AdaptiveState:
    def __init__(self):
        self.threshold = 0.1
        self.previous_mean_reward = None
        self.step = 0


STATE = AdaptiveState()


def normalize_groups(q2rollouts, index, standard_scores, raw_scores, *,
                     state=STATE, tau=1e-6, anchor=0.1, alpha=0.5,
                     eta=0.01, eps=1e-8):
    """Return per-question advantages and diagnostics; virtual samples have no gradient.

    ACR/threshold/K follow AVSPO equations 6, 8 and 10. The reward-B anchor
    adapts equation 9 to a continuous, composite reward. Only real answers
    receive advantages. Metadata must align with the rollout order.
    """
    positions = defaultdict(list)
    for pos, uid in enumerate(index):
        positions[uid].append(pos)
    if set(positions) != set(q2rollouts):
        raise ValueError("Question IDs do not match HARPO reward groups")
    if len(standard_scores) != len(index) or len(raw_scores) != len(index):
        raise ValueError("Reward and correctness metadata length mismatch")
    if not q2rollouts:
        raise ValueError("Empty rollout batch")
    collapsed = {uid: float(np.std(vals, ddof=0)) < tau
                 for uid, vals in q2rollouts.items()}
    acr = sum(collapsed.values()) / len(collapsed)
    trigger = acr > state.threshold
    k = max(1, min(max(len(v) for v in q2rollouts.values()),
                   math.ceil(max(len(v) for v in q2rollouts.values()) * acr**alpha)))
    normalized = {}
    stats = {"acr": acr, "threshold": state.threshold,
             "triggered": int(trigger), "collapsed_groups": sum(collapsed.values()),
             "all_wrong_groups": 0, "all_wrong_collapsed": 0,
             "all_wrong_noncollapsed": 0, "all_correct_collapsed": 0,
             "augmented_groups": 0, "virtual_count": 0, "k": k}
    for uid, vals in q2rollouts.items():
        vals = [float(value) for value in vals]
        positions_for_uid = positions[uid]
        if len(vals) != len(positions_for_uid) or len(vals) != 4:
            raise ValueError(f"Expected four aligned answers for {uid}")
        correctness = [float(standard_scores[i]) for i in positions_for_uid]
        if any(value not in (0.0, 1.0) for value in correctness):
            raise ValueError(f"Non-binary standard_score for {uid}: {correctness}")
        all_wrong = all(value == 0.0 for value in correctness)
        all_correct = all(value == 1.0 for value in correctness)
        stats["all_wrong_groups"] += int(all_wrong)
        stats["all_wrong_collapsed"] += int(all_wrong and collapsed[uid])
        stats["all_wrong_noncollapsed"] += int(all_wrong and not collapsed[uid])
        stats["all_correct_collapsed"] += int(all_correct and collapsed[uid])
        rewards = vals
        if trigger and collapsed[uid] and (all_wrong or all_correct):
            observed = vals[0]
            if all_wrong:
                virtual = [observed + anchor * (k - j) / k for j in range(k)]
            else:
                virtual = [observed - anchor * (j + 1) / (k + 1)
                           for j in range(k)]
            rewards = vals + virtual
            stats["augmented_groups"] += 1
            stats["virtual_count"] += k
        mu = float(np.mean(rewards))
        sd = float(np.std(rewards, ddof=0))
        normalized[uid] = [(value - mu) / max(sd, eps) for value in vals]
        if not all(math.isfinite(value) for value in normalized[uid]):
            raise FloatingPointError(f"Nonfinite AVSPO advantage for {uid}")
        if trigger and collapsed[uid] and all_wrong and not all(
                value < 0 for value in normalized[uid]):
            raise AssertionError("All-wrong virtual anchor did not yield negative advantage")
    mean_reward = sum(float(value) for value in raw_scores) / len(raw_scores)
    if state.previous_mean_reward is not None:
        delta = mean_reward - state.previous_mean_reward
        direction = (delta > 0) - (delta < 0)
        state.threshold += eta * direction * (acr - state.threshold)
        state.threshold = min(1.0, max(0.0, state.threshold))
    state.previous_mean_reward = mean_reward
    state.step += 1
    stats["step"] = state.step
    return normalized, stats
