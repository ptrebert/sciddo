# coding=utf-8

"""
The algorithms module contains
isolated implementations that
could be extracted and used
in other projects.
"""

import array as arr


def get_max_idx(scores, other):
    """
    Right to left (reversed) iteration through
    scores to find maximal index in scores
    where the item at index i is smaller than
    "other"

    :param scores:
    :param other:
    :return:
    """
    for idx, item in zip(range(-1, -len(scores) - 1, -1), reversed(scores)):
        if item < other:
            return idx, item
    return 0, None


def get_all_max_scoring_subseq(scores):
    """
    This implements the linear time algorithm described in:

    Ruzzo & Tompa, Proceedings of ISMB 1999 (PMID 10786306)

    The implementation is slightly different in that it tracks
    only start and end position of all maximal scoring subsequences
    instead of the sequences themselves.
    Note that by definition in the above paper,
    "a maximal scoring subsequence [is] any of the positive scoring
    subsequences found by [this algorithm]".
    Any filtering of the resulting set of sequences
    has to be done in a subsequent step.

    :param scores: scores describing characteristics of interest between two sequences of the same alphabet
     :type: iterable of numerics
    :return: all maximal scoring subsequences
     :rtype: list of tuple(cum. score,  start index, end index)
    """

    # Init: initialize tracking lists and cumulative sum counter
    # left and right scores plus start-end indices of subsequences
    seq_starts = arr.array('I', [])
    seq_ends = arr.array('I', [])
    scores_left = arr.array('f', [])
    scores_right = arr.array('f', [])
    cumsum = 0

    res_starts = []
    res_ends = []

    find_max_idx = get_max_idx

    # One-time iteration over score list
    # Following the notation in the above paper,
    # i_k is k-th (= current iteration) subsequence under consideration
    # i_j is some other previous subsequence
    # _left: cum. score up to, but not including the leftmost position of subsequence i
    # _right: cum. score up to and including the rightmost position of subsequence i
    for idx, score in enumerate(scores):
        if score > 0:
            # Left score does not include current position
            i_k_left = cumsum
            cumsum += score
            # Right score includes current position
            i_k_right = cumsum
            # This is a minor difference to the manuscript,
            # just keep track of start and end indices for each
            # subsequence instead of the sequence itself
            i_k_start = idx
            i_k_end = idx + 1
            while 1:
                try:
                    # The list of left scores is searched from
                    # right to left for the maximal index
                    # max(j): s.t. i_j_left < i_k_left
                    # STEP 1
                    j, i_j_left = find_max_idx(scores_left, i_k_left)
                    if not j:
                        # index position 0
                        # STEP 2
                        raise ValueError
                    # There is such j s.t. i_j_left < i_k_left, go on...
                    # This assert is unnecessary by implementation of get_max_idx
                    # assert i_j_left < i_k_left, '1st left score assumption false: {} !< {}'.format(i_j_left, i_k_left)
                    i_j_right = scores_right[j]
                    if i_j_right >= i_k_right:
                        # We cannot increase the score of the
                        # previous subsequence i_j by appending
                        # the current subsequence i_k to it.

                        # Record position of subsequence i_k
                        # and the respective left/right scores
                        # STEP 3
                        seq_starts.append(i_k_start)
                        seq_ends.append(i_k_end)
                        scores_left.append(i_k_left)
                        scores_right.append(i_k_right)
                        # subsequence i_k is processed, break while loop
                        break
                    else:
                        # Case: i_j_right < i_k_right
                        # We can increase the score of the
                        # previous subsequence i_j by appending
                        # the current subsequence i_k to it
                        # STEP 4
                        i_k_left = scores_left[j]
                        i_k_start = seq_starts[j]
                        seq_starts = seq_starts[:j]
                        seq_ends = seq_ends[:j]
                        scores_left = scores_left[:j]
                        scores_right = scores_right[:j]
                        # Use extended subsequence i_j
                        # and start from step 1 (while loop)
                        continue
                except ValueError:
                    # This happens, e.g., when processing
                    # the first subsequence and left_scores
                    # is still empty
                    # STEP 2

                    # Implementation note: the authors suggest
                    # to replace STEP 2 with STEP 2' which outputs
                    # all already identified segments (they are
                    # maximal for the prefix of the score list and
                    # thus are also maximal for the entire list)
                    # While their argument is w.r.t. memory requirements
                    # of the algorithm, this has no effect on this
                    # implementation, as segment indices are just copied
                    # to the result lists.
                    # However, this has the benefit that the scores_left
                    # list is always short and the runtime of get_max_idx
                    # is lower - mimicking the behavior of storing "pointers"
                    # of the identified segment j (Step 1) when adding a
                    # new segment I_k (Step 3) in Python seems not worth
                    # the effort
                    res_starts.extend(seq_starts)
                    res_ends.extend(seq_ends)

                    seq_starts = arr.array('I', [i_k_start])
                    seq_ends = arr.array('I', [i_k_end])
                    scores_left = arr.array('f', [i_k_left])
                    scores_right = arr.array('f', [i_k_right])
                    break
        else:
            # negative scores do not need
            # any special treatment
            cumsum += score

    if seq_starts:
        assert len(seq_starts) == len(seq_ends), \
            'Segment start/end pairs do not match: {} - {}'.format(seq_starts, seq_ends)
        res_starts.extend(seq_starts)
        res_ends.extend(seq_ends)
    res = sorted([(sum(scores[s:e]), s, e) for s, e in zip(res_starts, res_ends)],
                 key=lambda t: (t[0], -t[1]), reverse=True)
    return res
