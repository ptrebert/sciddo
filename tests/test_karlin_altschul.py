# coding=utf-8

import numpy as np
import pytest as pytest

from scidlib.statistics import estimate_ka_lambda, compute_ka_h, \
    init_karlin_altschul_estimation, normalize_scores


def test_karlin_python():
    """
    This function tests the Python
    implementation for lambda parameter
    estimation and subsequent calculation
    of the H (entropy) parameter.
    Estimation of the parameter K has not yet
    been re-implemented in Python

    Test values are taken from the original
    C source code (~1990) / the developer comment

    You can find a copy of that code / comment
    in the BLAST developer notes (Gertz, 2005)

    ftp://ftp.ncbi.nlm.nih.gov/blast/documents/developer/scoring.pdf

    :return:
    """
    score_values = [-2, -1, 0, 1, 2, 3]
    score_probs = [0.7, 0.0, 0.1, 0.0, 0.0, 0.2]
    # the expected values are of higher precision
    # than stated in the dev comment - this precision
    # is not needed for practical purposes, though
    lambda_expect = 0.32995
    h_expect = 0.29394

    lambda_est = estimate_ka_lambda(score_values, score_probs)
    assert np.isclose(lambda_expect, lambda_est, atol=0.001), \
        'Lambda estimate is off: est {} vs exp {}'.format(lambda_est, lambda_expect)

    h_est = compute_ka_h(lambda_est, score_values, score_probs)
    assert np.isclose(h_expect, h_est, atol=0.001), \
        'H estimate is off: est {} vs exp {}'.format(h_est, h_expect)
    return True


def test_karlin_module():
    """
    This function tests the C extension
    for Karlin-Altschul parameter estimation for
    lambda and subsequent calculation
    of the H (entropy) parameter and estimation
    of parameter K.

    Test values are taken from the original
    C source code (~1990) / the developer comment

    You can find a copy of that code / comment
    in the BLAST developer notes (Gertz, 2005)

    ftp://ftp.ncbi.nlm.nih.gov/blast/documents/developer/scoring.pdf

    :return:
    """
    import karlin as karlin
    score_values = [-2, -1, 0, 1, 2, 3]
    score_probs = [0.7, 0.0, 0.1, 0.0, 0.0, 0.2]
    # the expected values are of higher precision
    # than stated in the dev comment - this precision
    # is not needed for practical purposes, though
    lambda_expect = 0.32995
    h_expect = 0.29394
    k_expect = 0.15399

    lambda_est, h_est, k_est = karlin.estimateParameters(min(score_values),
                                                         max(score_values),
                                                         score_probs)

    assert np.isclose(lambda_expect, lambda_est, atol=0.001), \
        'Lambda estimate is off: est {} vs exp {}'.format(lambda_est, lambda_expect)

    assert np.isclose(h_expect, h_est, atol=0.001), \
        'H estimate is off: est {} vs exp {}'.format(h_est, h_expect)

    assert np.isclose(k_expect, k_est, atol=0.001), \
        'K estimate is off: est {} vs exp {}'.format(k_est, k_expect)

    return True


def test_karlin_c_source():
    """
    This function tests the C source
    implementation for lambda parameter
    estimation and subsequent calculation
    of the H (entropy) parameter and estimation
    of parameter K.

    Test values are taken from the original
    C source code (~1990) / the developer comment

    You can find a copy of that code / comment
    in the BLAST developer notes (Gertz, 2005)

    ftp://ftp.ncbi.nlm.nih.gov/blast/documents/developer/scoring.pdf

    :return:
    """
    import re
    with pytest.raises(RuntimeError, match=re.escape('karlin module can be imported.')):
        est_ka_params = init_karlin_altschul_estimation()

    return True


def test_score_normalization():
    """
    This tests score normalization with values
    taken from Karlin & Altschul, PNAS 1993, 10.1073/pnas.90.12.5873

    :return:
    """
    # Example Table 1
    raw_scores = [67, 55, 38]
    k = 0.21
    l = 0.159
    mn = 2594
    norm_scores = [4.4, 2.5, -0.2]

    # NB:
    # yes, the tolerance is chosen to fit the expectation,
    # the numbers given in the paper are heavily rounded...

    calc_scores = normalize_scores(raw_scores, l, k, 'e', mn)
    assert np.allclose(norm_scores, calc_scores, atol=0.06), \
        'Normalized scores are wrong for example Table 1: {} vs {}'.format(norm_scores, calc_scores)

    # Example Table 3
    raw_scores = np.array([52, 49, 46], dtype=np.float16)
    k = 0.17
    l = 0.314
    mn = 34336
    # vary the object type to be sure it works
    norm_scores = np.array([7.6, 6.7, 5.8], dtype=np.float16)

    calc_scores = normalize_scores(raw_scores, l, k, 'ln', mn)
    assert np.allclose(norm_scores, calc_scores, atol=0.06), \
        'Normalized scores are wrong for example Table 3: {} vs {}'.format(norm_scores, calc_scores)

    return True
