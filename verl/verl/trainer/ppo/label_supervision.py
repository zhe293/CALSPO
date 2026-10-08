"""Conditional label-only teacher forcing; no generated reasoning is supervised."""
from collections import defaultdict
import math
import os
import json
import numpy as np
import torch

LABELS = {
    'emotion_cls': {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'},
    'depression_cls': {'depression', 'no depression'},
}
CLASS_WEIGHTS = json.loads(os.environ.get('HBA_LABEL_CLASS_WEIGHTS', '{}'))

KEYS = ('label_sft_selected', 'label_sft_targets', 'label_sft_mask', 'label_sft_content_mask', 'label_sft_error_weight')


def attach_targets(batch, tokenizer, group_size=4):
    """Select one row per UID regardless of sampled correctness, before actor minibatch splitting."""
    scores = np.asarray(batch.non_tensor_batch['standard_score'], dtype=float)
    uids = batch.non_tensor_batch['uid']
    if len(scores) != len(uids) or not np.isfinite(scores).all():
        raise ValueError('Invalid correctness metadata')
    if not np.isin(scores, [0.0, 1.0]).all():
        raise ValueError('standard_score must be binary label correctness')
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    n = len(uids)
    selected = torch.zeros(n, dtype=torch.bool, device=batch.batch['responses'].device)
    encoded = {}
    content_masks = {}
    weights = torch.zeros(n, dtype=torch.float32, device=selected.device)
    error_histogram = defaultdict(int)
    by_task = defaultdict(int)
    for uid, rows in groups.items():
        if len(rows) != group_size:
            raise ValueError(f'Expected {group_size} answers per UID; got {len(rows)}')
        errors = int(np.count_nonzero(scores[rows] == 0))
        error_histogram[errors] += 1
        labels = [str(batch.non_tensor_batch['reward_model'][i]['ground_truth']).strip().lower()
                  for i in rows]
        tasks = [str(batch.non_tensor_batch['task'][i]) for i in rows]
        if len(set(labels)) != 1 or len(set(tasks)) != 1:
            raise ValueError('Inconsistent task/label within question group')
        label, task = labels[0], tasks[0]
        if task not in LABELS or label not in LABELS[task]:
            raise ValueError(f'Unsupported supervised target: {task}/{label}')
        # The response is the fixed answer template, followed by EOS; no <think> text.
        text = '\\boxed{' + label + '}'
        tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenized['input_ids']
        offsets = tokenized['offset_mapping']
        start = len('\\boxed{')
        end = start + len(label)
        content = [b > start and a < end for a, b in offsets]
        if not any(content):
            raise ValueError('Label tokens missing from tokenizer offsets')
        if not tokens or tokenizer.eos_token_id is None:
            raise ValueError('Empty target or missing EOS')
        tokens = tokens + [tokenizer.eos_token_id]
        if len(tokens) > batch.batch['responses'].shape[-1]:
            raise ValueError('Label target exceeds response limit')
        selected[rows[0]] = True
        weights[rows[0]] = float(CLASS_WEIGHTS.get(task + '/' + label, 1.0))
        encoded[rows[0]] = tokens
        content_masks[rows[0]] = content + [False]
        by_task[task] += 1
    width = max([len(v) for v in encoded.values()] or [1])
    targets = torch.full((n, width), tokenizer.pad_token_id, dtype=torch.long,
                         device=selected.device)
    mask = torch.zeros((n, width), dtype=torch.bool, device=selected.device)
    content_mask = torch.zeros_like(mask)
    for i, tokens in encoded.items():
        targets[i, :len(tokens)] = torch.tensor(tokens, device=selected.device)
        mask[i, :len(tokens)] = True
        content_mask[i, :len(tokens)] = torch.tensor(content_masks[i], device=selected.device)
    batch.batch['label_sft_selected'] = selected
    batch.batch['label_sft_error_weight'] = weights
    batch.batch['label_sft_targets'] = targets
    batch.batch['label_sft_mask'] = mask
    batch.batch['label_sft_content_mask'] = content_mask
    stats = {'groups': len(encoded), 'by_task': dict(by_task),
             'target_tokens': int(mask.sum()), 'total_questions': len(groups), 'error_histogram': dict(error_histogram),
             'error_weight_sum': float(weights.sum())}
    print(f'[LABEL_SFT_TARGETS] {stats}', flush=True)
    return stats


def teacher_inputs(inputs):
    """Retain original prompt IDs, masks, multimodal tensors and mRoPE prompt positions."""
    rows = torch.where(inputs['label_sft_selected'])[0]
    if rows.numel() == 0:
        return None
    target = inputs['label_sft_targets'][rows]
    mask = inputs['label_sft_mask'][rows]
    prompt_len = inputs['input_ids'].shape[-1] - inputs['responses'].shape[-1]
    if prompt_len <= 0:
        raise ValueError('Missing prompt prefix')
    prompt_ids = inputs['input_ids'][rows, :prompt_len]
    prompt_attention = inputs['attention_mask'][rows, :prompt_len]
    prompt_positions = inputs['position_ids'][rows, ..., :prompt_len]
    delta = torch.arange(1, target.shape[-1]+1, device=target.device)
    suffix_positions = prompt_positions[..., -1:] + delta
    result = {
        'input_ids': torch.cat([prompt_ids, target], dim=-1),
        'attention_mask': torch.cat([prompt_attention, mask.to(prompt_attention.dtype)], dim=-1),
        'position_ids': torch.cat([prompt_positions, suffix_positions], dim=-1),
        'responses': target,
        'response_mask': mask,
        'label_content_mask': inputs['label_sft_content_mask'][rows],
        'error_weight': inputs['label_sft_error_weight'][rows],
    }
    if 'multi_modal_inputs' in inputs:
        indices = rows.detach().cpu().tolist()
        result['multi_modal_inputs'] = np.array(
            [inputs['multi_modal_inputs'][i] for i in indices], dtype=object)
    return result


def supervised_backward(actor, inputs, denominator, coefficient):
    """Accumulate auxiliary gradients with the same scaler and optimizer as PPO.

    denominator is the count of distinct selected questions in this PPO minibatch.
    NLL averages label tokens and format/EOS tokens separately per question.
    Format/EOS weight is 0.5; train-frequency class weight multiplies each question; selected questions are averaged per PPO minibatch.
    Called after the PPO backward, before its optimizer step to limit peak memory.
    """
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Invalid supervision coefficient')
    teacher = teacher_inputs(inputs)
    if teacher is None:
        return {'actor/label_sft_groups': 0.0, 'actor/label_sft_loss': 0.0}
    if denominator <= 0:
        raise ValueError('Invalid SFT question denominator')
    _, log_probs = actor._forward_micro_batch(teacher, temperature=1.0, calculate_entropy=False)
    content = teacher['label_content_mask']
    formatting = teacher['response_mask'] & ~content
    token_nll = -log_probs.float()
    label_nll = token_nll.masked_fill(~content, 0.0).sum(-1) / content.sum(-1).clamp_min(1)
    format_nll = token_nll.masked_fill(~formatting, 0.0).sum(-1) / formatting.sum(-1).clamp_min(1)
    w = teacher['error_weight']
    if not torch.isfinite(w).all() or not ((w >= 0.5) & (w <= 2)).all():
        raise ValueError('Invalid class weight')
    nll = (w * (label_nll + 0.5 * format_nll)).sum() / denominator
    if not torch.isfinite(nll):
        raise FloatingPointError('Nonfinite label supervision NLL')
    weighted = coefficient * nll
    actor._p40_scaler.scale(weighted).backward()
    stats = {'actor/label_sft_content_loss': float((w * label_nll).detach().sum()/denominator),
             'actor/label_sft_format_loss': float((w * format_nll).detach().sum()/denominator),
             'actor/label_sft_format_weight': 0.5,
             'actor/label_sft_error_weight_sum': float(w.sum()),
             'actor/label_sft_groups': float(teacher['responses'].shape[0]),
             'actor/label_sft_loss': float(nll.detach()),
             'actor/label_sft_weighted_loss': float(weighted.detach()),
             'actor/label_sft_tokens': float(teacher['response_mask'].sum())}
    print(f'[LABEL_SFT_BACKWARD] {stats}', flush=True)
    return stats
