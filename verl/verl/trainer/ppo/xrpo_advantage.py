"""XRPO changes 2+3 on the repository HARPO; no replacement algorithm."""
from types import SimpleNamespace
STATE = SimpleNamespace(last_metrics={})

def compute(token_level_rewards, response_mask, index, task_ids,
            dataset_ids, class_labels, *, mode):
    if mode != "xrpo_23":
        raise ValueError("Only the approved repository-based xrpo_23 is supported")
    from verl.trainer.ppo import core_algos
    return core_algos.compute_harpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=response_mask,
        index=index, task_ids=task_ids, dataset_ids=dataset_ids, class_labels=class_labels,
        beta_mu=0.95, beta_sigma=0.95, beta_mean=0.95, beta_cvar=0.95,
        beta_tail=1.0, class_count_info=None, class_weight_scope="auto",
        xrpo_beta=0.90, xrpo_reverse_reference=True, xrpo_metrics=STATE.last_metrics,
    )
