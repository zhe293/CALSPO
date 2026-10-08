"""Preserve HARPO grouping metadata after preparing a generation batch.

The generation worker receives task/class_label through gen_batch, while
compute_advantage later reads them from the training batch. Keep a separate
copy in the training batch before rollout repetition.
"""
from pathlib import Path

path = Path(__file__).resolve().parents[2] / 'verl/verl/trainer/ppo/ray_trainer.py'
text = path.read_text()
needle = '''                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                # pass global_steps to trace
'''
replacement = '''                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                # HARPO needs these identifiers after generation to group the
                # repeated responses by task and class during advantage calculation.
                if self.config.algorithm.adv_estimator == AdvantageEstimator.HARPO:
                    for key in ("task", "class_label"):
                        if key not in gen_batch.non_tensor_batch:
                            raise KeyError(f"HARPO requires {key} in the training batch")
                        batch.non_tensor_batch[key] = gen_batch.non_tensor_batch[key].copy()
                # pass global_steps to trace
'''
if replacement in text:
    print('harpo_metadata_fix=already_applied')
else:
    if text.count(needle) != 1:
        raise RuntimeError('Expected generation split not found exactly once')
    path.write_text(text.replace(needle, replacement))
    print('harpo_metadata_fix=applied')
