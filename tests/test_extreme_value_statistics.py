# coding=utf-8

import numpy as np

from scidlib.statistics import single_segment_score_pv, \
    multi_segment_score_pv, summed_score_pv


def test_single_segment_pv():
    """
    Test values taken from
    Karlin & Altschul, PNAS 1993, 10.1073/pnas.90.12.5873

    :return:
    """
    # Values from Table 1
    norm_scores = [4.4, 2.5, -0.2]
    pv_expect = [0.012, 0.08, 0.71]

    # need to cast to float here since, in raw mode, the p-value
    # functions return an mpf object
    pv_calc = list(map(float, list(map(single_segment_score_pv, norm_scores))))
    assert np.allclose(pv_expect, pv_calc, atol=0.01), \
        'Computed p-values for example Table 1 are wrong: {} vs {}'.format(pv_expect, pv_calc)

    # Values from Table 3
    norm_scores = [7.6, 6.7, 5.8]
    pv_expect = [0.00047, 0.0012, 0.0031]

    pv_calc = list(map(float, list(map(single_segment_score_pv, norm_scores))))
    assert np.allclose(pv_expect, pv_calc, atol=0.01), \
        'Computed p-values for example Table 3 are wrong: {} vs {}'.format(pv_expect, pv_calc)

    return True


def test_summed_score_pv():
    """
    Test values taken from
    Karlin & Altschul, PNAS 1993, 10.1073/pnas.90.12.5873

    :return:
    """
    # Values from Table 3
    norm_scores = [7.6, 6.7, 5.8]
    pv_expect = [0.00047, 0.0000042, 0.000000059]

    pv_calc = [float(summed_score_pv(norm_scores[0], 1)),
               float(summed_score_pv(np.sum(norm_scores[:2]), 2)),
               float(summed_score_pv(np.sum(norm_scores), 3))]
    assert np.allclose(pv_expect, pv_calc, atol=0.0001), \
        'Computed p-values for example Table 3 are wrong: {} vs {}'.format(pv_expect, pv_calc)

    return True
