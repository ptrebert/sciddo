# coding=utf-8

from scidlib.algorithms import get_all_max_scoring_subseq


def test_ruzzo_tompa_algorithm():
    """
    Test values are taken from
    Ruzzo & Tompa, ISMB Proceedings 1999

    :return:
    """
    # Testing Ruzzo & Tompa algorithm using the example
    # provided in the manuscript PMID: 10786306
    input_scores = [4, -5, 3, -3, 1, 2, -2, 2, -2, 1, 5]
    output_segments = [(7, 4, 11), (4, 0, 1), (3, 2, 3)]
    num_expected = 3

    results = get_all_max_scoring_subseq(input_scores)
    num_res = len(results)
    assert num_res == num_expected, \
        'Result list differs from expected length: {} vs {}'.format(num_res, num_expected)
    assert all([s in output_segments for s in results]), \
        'Not all segments are contained in the expected solution set: {} vs {}'.format(results, output_segments)
    return True
