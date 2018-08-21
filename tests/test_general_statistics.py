# coding=utf-8

import pandas as pd
import numpy as np

from scidlib.statistics import jensen_shannon_divergence as jsd
from scidlib.statistics import compute_score_probabilities


def test_js_divergence():
    """
    The non-trivial case has been confirmed
    with the R philentropy package v0.1.0 / R v3.4.1

    :return:
    """
    p = np.array([0.00029421, 0.42837957, 0.1371827, 0.00029419, 0.00029419,
                  0.40526004, 0.02741252, 0.00029422, 0.00029417, 0.00029418], dtype=np.float32)
    q = np.array([0.00476199, 0.004762, 0.004762, 0.00476202, 0.95714168,
                  0.00476213, 0.00476212, 0.00476202, 0.00476202, 0.00476202], dtype=np.float32)
    jsd_calc = jsd(p, q)
    jsd_exp = 0.9316258
    assert np.isclose(jsd_calc, jsd_exp, atol=10e-6), 'JSD-1 failed: exp {} vs calc {}'.format(jsd_exp, jsd_calc)

    # JSD(P, P) has to be 0.
    p = np.array([0.00029421, 0.42837957, 0.1371827, 0.00029419, 0.00029419,
                  0.40526004, 0.02741252, 0.00029422, 0.00029417, 0.00029418], dtype=np.float32)
    q = p
    jsd_calc = jsd(p, q)
    jsd_exp = 0.
    assert np.isclose(jsd_calc, jsd_exp, atol=10e-6), 'JSD-2 failed: exp {} vs calc {}'.format(jsd_exp, jsd_calc)

    # re-scaling should not affect the results
    p = np.array([0.00029421, 0.42837957, 0.1371827, 0.00029419, 0.00029419,
                  0.40526004, 0.02741252, 0.00029422, 0.00029417, 0.00029418], dtype=np.float32)
    p *= 10
    q = np.array([0.00476199, 0.004762, 0.004762, 0.00476202, 0.95714168,
                  0.00476213, 0.00476212, 0.00476202, 0.00476202, 0.00476202], dtype=np.float32)
    q *= 10
    jsd_calc = jsd(p, q)
    jsd_exp = 0.9316258
    assert np.isclose(jsd_calc, jsd_exp, atol=10e-6), 'JSD-3 failed: exp {} vs calc {}'.format(jsd_exp, jsd_calc)

    return True


def test_compute_score_probabilities():
    """
    :return:
    """
    score_matrix = pd.DataFrame([[0, 1, 3],
                                 [1, 2, 1],
                                 [3, 1, 1]],
                                index=[1, 2, 3],
                                columns=[1, 2, 3])
    # the index of the Series is deliberately not sorted
    # to implicitly test that label-based and index-based
    # lookup are not mixed
    base_freq = pd.Series([0.3, 0.2, 0.5], index=[2, 1, 3])

    expected_range = [0, 1, 2, 3]
    expected_probs = [0.04, 0.67, 0.09, 0.2]

    computed_range, computed_probs = compute_score_probabilities(score_matrix, base_freq)

    assert np.allclose(expected_range, computed_range, atol=1e-6), \
        'Computed score range wrong: {} vs {}'.format(expected_range, computed_range)
    assert np.allclose(expected_probs, computed_probs, atol=1e-3), \
        'Computed score probabilities wrong: {} vs {}'.format(expected_probs, computed_probs)

    return True
