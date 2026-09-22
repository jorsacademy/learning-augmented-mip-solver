import math

from scipy.stats import t

from ltb_milp.statistics import (
    paired_deltas,
    paired_percent_reductions,
    summarize_samples,
)


def test_summarize_samples_computes_student_t_ci() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(summary.mean, 2.5)
    assert math.isclose(summary.std, 1.2909944487358056)
    expected = float(t.ppf(0.975, df=3)) * summary.std / 2.0
    assert math.isclose(summary.ci95, expected)
    assert summary.n == 4


def test_summarize_samples_single_value_has_zero_uncertainty() -> None:
    summary = summarize_samples([7.0])
    assert summary.mean == 7.0
    assert summary.std == 0.0
    assert summary.ci95 == 0.0
    assert summary.n == 1


def test_summarize_samples_rejects_empty_input() -> None:
    try:
        summarize_samples([])
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_paired_deltas_preserve_pairing() -> None:
    assert paired_deltas([10.0, 20.0], [8.0, 25.0]) == [-2.0, 5.0]


def test_paired_percent_reductions_positive_means_improvement() -> None:
    reductions = paired_percent_reductions([10.0, 20.0], [8.0, 15.0])
    assert reductions == [20.0, 25.0]


def test_paired_helpers_validate_input() -> None:
    for function in (paired_deltas, paired_percent_reductions):
        try:
            function([1.0], [1.0, 2.0])
        except ValueError as exc:
            assert "identical lengths" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    try:
        paired_percent_reductions([0.0], [0.0])
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
