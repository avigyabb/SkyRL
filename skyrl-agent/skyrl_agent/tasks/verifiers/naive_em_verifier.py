def compute_score(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """Compute the reward score for a solution by exact string match
    """
    del extra_info # unused
    # First assert intended generation and gt type
    model_output = str(solution_str)
    ground_truth = str(ground_truth)

    if model_output == ground_truth:
        correct = True
    else:
        correct = False

    # reward = 1.0 if correct else -1.0
    reward = 1.0 if correct else 0.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
    }
    
