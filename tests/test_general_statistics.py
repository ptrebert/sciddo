# coding=utf-8

import pandas as pd
import numpy as np

from scidlib.statistics import jensen_shannon_divergence as jsd
from scidlib.statistics import compute_score_probabilities, compute_scan_parameters
from scidlib.cmd_scan import compute_adaptive_group_size


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


def test_adaptive_length_norm_factor():
    """
    :return:
    """
    g1 = [['A', 'B', 'B'], ['A', 'B', 'C'], ['A', 'B', 'A']]
    g1 = [pd.Series(i) for i in g1]
    g2 = [['A', 'B', 'C'], ['A', 'B', 'C']]
    g2 = [pd.Series(i) for i in g2]

    g1_size = compute_adaptive_group_size(g1, len(g1[0]))
    g2_size = compute_adaptive_group_size(g2, len(g2[0]))

    assert g1_size == 5, 'Group 1 size is wrong: {} vs expected {}'.format(g1_size, 5)
    assert g2_size == 3, 'Group 2 size is wrong: {} vs expected {}'.format(g2_size, 3)

    chroms = pd.DataFrame([[3, 1]],
                          index=['chrT'],
                          columns=['bins', 'binsize'])

    params = pd.DataFrame([[0.5, 0.75, 0.8]],
                          index=['chrT'],
                          columns=['ka_k', 'ka_lambda', 'ka_h'])

    group1 = {'total': 3, 'singletons': 0, 'rep_groups': 1}
    group2 = {'total': 2, 'singletons': 0, 'rep_groups': 1}
    baselength = {'chrT': 3}
    grouplength = pd.DataFrame([[g1_size, g2_size]],
                               index=['chrT'],
                               columns=['group1', 'group2'])

    res = compute_scan_parameters(chroms, params, group1, group2, baselength, grouplength, None)

    ln = res['chrT']['len_norm']
    assert ln == 'full_adaptive', 'Unexpected length normalization factor computed: {}'.format(ln)

    g1_len = res['chrT']['eff_grp1_len']
    assert g1_len == 2, 'Unexpected group 1 length: {} vs expected {}'.format(g1_len, 2)
    g2_len = res['chrT']['eff_grp2_len']
    assert g2_len == 3, 'Unexpected group 2 length: {} vs expected {}'.format(g2_len, 3)

    return True


def test_linear_length_norm_factor():
    """
    :return:
    """
    g1 = [['A', 'B', 'B'], ['A', 'B', 'C'], ['A', 'B', 'A']]
    g2 = [['A', 'B', 'C'], ['A', 'B', 'C']]

    chroms = pd.DataFrame([[3, 1]],
                          index=['chrT'],
                          columns=['bins', 'binsize'])

    params = pd.DataFrame([[0.5, 0.75, 0.8]],
                          index=['chrT'],
                          columns=['ka_k', 'ka_lambda', 'ka_h'])

    group1 = {'total': 3, 'singletons': 0, 'rep_groups': 1}
    group2 = {'total': 2, 'singletons': 0, 'rep_groups': 1}
    baselength = {'chrT': 3}
    grouplength = 'linear'

    res = compute_scan_parameters(chroms, params, group1, group2, baselength, grouplength, None)

    ln = res['chrT']['len_norm']
    assert ln == 'full_linear', 'Unexpected length normalization factor computed: {}'.format(ln)

    g1_len = res['chrT']['eff_grp1_len']
    assert g1_len == 18, 'Unexpected group 1 length: {} vs expected {}'.format(g1_len, 18)
    g2_len = res['chrT']['eff_grp2_len']
    assert g2_len == 0, 'Unexpected group 2 length: {} vs expected {}'.format(g2_len, 0)

    return True
  