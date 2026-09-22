from learning_cp_sat.statistics import paired_difference, summarize


def test_summary_and_paired_difference_schema() -> None:
    summary = summarize([1, 2, 3, 4])
    assert summary["n"] == 4
    assert summary["mean"] == 2.5

    paired = paired_difference([2, 4, 6], [1, 5, 3], bootstrap_seed=9, bootstrap_samples=200)
    assert paired["n"] == 3
    assert paired["ci95_low"] <= paired["mean_difference"] <= paired["ci95_high"]
