from original_reward import score

def human_behaviour_compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    return score('b', data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
