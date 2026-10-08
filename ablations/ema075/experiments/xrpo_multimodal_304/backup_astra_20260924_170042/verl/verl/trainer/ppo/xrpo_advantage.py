"""Paper-style HARPO and the proposed changes 2+3 for the 304-row study.

This is deliberately separate from the repository fork's historical HARPO code.
Both study arms use this implementation; only beta and the added x_m differ.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass
class AdvantageState:
    question_signal: dict[tuple[str, str], float] = field(default_factory=dict)
    question_scale: dict[tuple[str, str], float] = field(default_factory=dict)
    task_signal: dict[str, float] = field(default_factory=dict)
    task_scale: dict[str, float] = field(default_factory=dict)
    task_question_reference: dict[str, float] = field(default_factory=dict)
    last_metrics: dict[str, float] = field(default_factory=dict)


STATE = AdvantageState()


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("Geometric reference needs at least one value")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _smooth(old: float | None, current: float, beta: float) -> float:
    return current if old is None else beta * old + (1 - beta) * current


def _smooth_scale(old: float | None, candidate: float, beta: float) -> float:
    return candidate if old is None else old**beta * candidate ** (1 - beta)


def compute(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    task_ids,
    *,
    mode: str,
    state: AdvantageState = STATE,
    eps: float = 1e-6,
    signal_floor: float = 1e-4,
    scale_min: float = 1 / 3,
    scale_max: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one scalar advantage per rollout, broadcast over valid tokens.

    `paper_harpo` and `xrpo_23` share the exact same reward normalization and
    task reference population. A persistent task reference is needed because
    HBA's batches are homogeneous by modality, often containing one task.
    """
    if mode not in {"paper_harpo", "xrpo_23"}:
        raise ValueError(f"Unknown advantage mode: {mode}")
    beta = 0.95 if mode == "paper_harpo" else 0.90
    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError("Expected matching [rollout, response_token] reward and mask tensors")
    count = token_level_rewards.shape[0]
    if len(index) != count or task_ids is None or len(task_ids) != count:
        raise ValueError("Each rollout must carry one question id and task id")

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1).float()
        groups: dict[str, list[int]] = defaultdict(list)
        group_tasks: dict[str, str] = {}
        for i in range(count):
            qid, task = str(index[i]), str(task_ids[i])
            if qid in group_tasks and group_tasks[qid] != task:
                raise ValueError("A rollout group contains different tasks")
            groups[qid].append(i)
            group_tasks[qid] = task

        base = torch.zeros_like(scores)
        p_question: dict[str, float] = {}
        for qid, positions in groups.items():
            if len(positions) < 2:
                raise ValueError(f"Group {qid} has fewer than two rollouts")
            values = scores[positions]
            advantages = (values - values.mean()) / (values.std(unbiased=False) + eps)
            base[positions] = advantages
            p_question[qid] = max(float(advantages.abs().mean()), signal_floor)

        task_groups: dict[str, list[str]] = defaultdict(list)
        for qid, task in group_tasks.items():
            task_groups[task].append(qid)

        # A question ID is stable only if supplied by the loader. On the first
        # observation, initialize its EMA from the observation itself.
        for qid, task in group_tasks.items():
            key = (task, qid)
            state.question_signal[key] = _smooth(
                state.question_signal.get(key), p_question[qid], beta
            )

        current_references: dict[str, float] = {}
        for task, qids in task_groups.items():
            current_references[task] = _geomean(
                [state.question_signal[(task, qid)] for qid in qids]
            )
            positions = [i for qid in qids for i in groups[qid]]
            task_observation = max(float(base[positions].abs().mean()), signal_floor)
            state.task_signal[task] = _smooth(
                state.task_signal.get(task), task_observation, beta
            )
            # A second task-level EMA makes R_m comparable across consecutive
            # single-modality batches. It is used by x_m, not by the paper arm.
            state.task_question_reference[task] = _smooth(
                state.task_question_reference.get(task), current_references[task], beta
            )

        task_ref = _geomean(list(state.task_signal.values()))
        question_ref_global = _geomean(list(state.task_question_reference.values()))
        final = base.clone()
        metrics: dict[str, float] = {
            "xrpo/seen_tasks": float(len(state.task_signal)),
            "xrpo/beta": beta,
            "xrpo/zero_groups": float(sum(p <= signal_floor for p in p_question.values())),
        }
        for task, qids in task_groups.items():
            task_candidate = task_ref / state.task_signal[task]
            task_scale = _smooth_scale(state.task_scale.get(task), task_candidate, beta)
            task_scale = max(scale_min, min(scale_max, task_scale))
            state.task_scale[task] = task_scale
            x = (
                state.task_question_reference[task] / question_ref_global
                if mode == "xrpo_23" else 1.0
            )
            x = max(scale_min, min(scale_max, x))
            safe_task = task.replace("/", "_")
            metrics[f"xrpo/task/{safe_task}/s_task"] = task_scale
            metrics[f"xrpo/task/{safe_task}/x"] = x
            metrics[f"xrpo/task/{safe_task}/combined"] = task_scale * x
            for qid in qids:
                key = (task, qid)
                question_candidate = current_references[task] / state.question_signal[key]
                question_scale = _smooth_scale(
                    state.question_scale.get(key), question_candidate, beta
                )
                question_scale = max(scale_min, min(scale_max, question_scale))
                state.question_scale[key] = question_scale
                final[groups[qid]] *= question_scale * task_scale * x

        if not torch.isfinite(final).all():
            raise FloatingPointError("Non-finite XRPO advantages")
        state.last_metrics = metrics
        broadcast = final.to(token_level_rewards.dtype).unsqueeze(-1) * response_mask
        return broadcast, broadcast
