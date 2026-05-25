def pie_score(compile_success, mean_acc, agg_runtime):
    if not compile_success:
        return 0
    return mean_acc * 0.5 + 1 / agg_runtime * 0.5
