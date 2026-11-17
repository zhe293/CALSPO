"""Use the author's reward B with a token-length cap matching this study."""

from original_reward import score


def human_behaviour_compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    if "response_lengths" not in kwargs:
        raise ValueError("Reward B requires tokenizer-measured response lengths")
    kwargs["max_response_length"] = 256
    kwargs["overlong_buffer_length"] = 64
    return score("b", data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
