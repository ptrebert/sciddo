# coding=utf-8

import os as os
import sys as sys
import numpy as np
import numpy.linalg as linalg
import scipy.optimize as opt
import scipy.stats as stats
import ctypes as cty
import itertools as itt
import collections as col

import pandas as pd
import mpmath as mpm

from scidlib import NUM_MAX_SIZE_SCORE, NUM_PREC_KA_PV


def jensen_shannon_divergence(p, q, base=2):
    """
    Compute Jensen-Shannon divergence between probability
    distributions p and q (default base 2 guarantees that
    divergence is bounded between 0 and 1).

    :param p:
    :param q:
    :return:
    """
    p_norm = p / linalg.norm(p, ord=1)
    q_norm = q / linalg.norm(q, ord=1)
    mean_dist = 0.5 * (p_norm + q_norm)
    jsd = 0.5 * (stats.entropy(p_norm, mean_dist, base=base) + stats.entropy(q_norm, mean_dist, base=base))
    return jsd


def compute_score_probabilities(score_mat, base_freq):
    """
    :param score_mat: Pandas.DataFrame of scores
    :param base_freq: Pandas.Series of base frequencies for
                      characters / state
    :return:
    """
    # Here, we need some type assertion since looking up
    # the state pairs uses the state numbers (integers), this
    # could lead to horrible off-by-X errors if the passed
    # structure is a numpy array with compatible dimensions
    assert isinstance(score_mat, pd.DataFrame), \
        'Score matrix object is not a Pandas DataFrame: {}'.format(type(score_mat))
    assert isinstance(base_freq, pd.Series), \
        'Base frequencies object is not a Pandas Series: {}'.format(type(base_freq))
    assert score_mat.index.isin(base_freq.index).all(), \
        'Not all state numbers are found in both object indices'
    collect_probs = col.Counter()
    # since the score matrix is symmetric by construction,
    # simple iterations through all (non-repeating)
    # combinations is enough
    for s1, s2 in itt.combinations_with_replacement(score_mat.index.tolist(), 2):
        score = score_mat.at[s1, s2]
        if s1 == s2:
            base_prob = 0.5 * base_freq.at[s1] * base_freq.at[s2]
        else:
            base_prob = base_freq.at[s1] * base_freq.at[s2]
        collect_probs[score] += base_prob
    score_range = sorted(collect_probs.keys())
    score_range = np.arange(score_range[0], score_range[-1] + 1, dtype=eval(NUM_MAX_SIZE_SCORE))
    assert score_range.min() == score_mat.min(axis=1).min(), 'Min score mismatch'
    assert score_range.max() == score_mat.max(axis=1).max(), 'Max score mismatch'
    prob_sum = sum(collect_probs.values())
    score_probs = np.array([collect_probs[s] / prob_sum for s in score_range], dtype=np.float32)
    assert np.isclose(score_probs.sum(), 1., atol=1e-6), \
        'Score probabilities do not sum to 1: {}'.format(score_probs.sum())
    return score_range, score_probs


def _get_log_function(unit):
    """
    :param unit:
    :return:
    """
    loglut = {'log2': np.log2, 'lb': np.log2, '2': np.log, 'bits': np.log2,
              'log10': np.log10, '10': np.log10, 'lg': np.log10,
              'loge': np.log, 'ln': np.log, 'e': np.log, 'nats': np.log}
    try:
        logfun = loglut[unit]
    except KeyError:
        loglist = ' - '.join(sorted(loglut.keys()))
        raise KeyError('The log function "{}" is not supported -'
                       ' please select from the following list: {}'.format(unit, loglist))
    return logfun


def normalize_scores(score, ka_lambda, ka_k, units, m, n=1):
    """
    Implements normalization of alignment score(s)
    as described in Karlin & Altschul, PNAS 1993

    S' = lambda * S - log ( K * N )

    For local alignment (pairwise) scoring, N has to
    be adjusted to N = N * M, where N and M are
    the lengths of the respective sequences being compared

    :param score:
    :param ka_lambda: Karlin-Altschul parameter lambda
    :param ka_k: Karlin-Altschul parameter K
    :param units: specify which log to use
     :type: str
    :param m: length of query sequence
    :param n: length of database sequence
    :return:
    """
    logfun = _get_log_function(units)
    # for linear length normalization, n or m could
    # be set to zero
    m = max(m, 1)
    n = max(n, 1)
    log_const = logfun(ka_k) + logfun(n) + logfun(m)
    if hasattr(score, '__iter__'):
        if isinstance(score, np.ndarray):
            # can use vectorized operations
            res = np.array(ka_lambda * score - log_const, dtype=np.float64)
        elif isinstance(score, pd.Series):
            res = pd.Series(ka_lambda * score - log_const, dtype=np.float64)
        else:
            # "compatibility" option
            # could be Python list or something else...
            res = list(map(lambda s: ka_lambda * s - log_const, score))
    else:
        # for single score value
        res = ka_lambda * score - log_const
    return res


def compute_expected_hsp_length(unit, ka_k, ka_h, m, sm, n, sn):
    """
    This computes the length of an HSP that has
    an expect value of 1 (called the expected
    HSP length)

    :param unit: log to use
    :param ka_k: Karlin-Altschul parameter K
    :param m: length of the query sequence
    :param sm: number of sequences in group 1
    :param n: length of the database
    :param sn: number of sequences in group 2
    :param ka_h: entropy of the scoring system
    :return: expected HSP length, effective length group1, effective length group2
    """
    logfun = _get_log_function(unit)
    log_const = logfun(ka_k) + logfun(n) + logfun(m)
    # This bound is described in BLAST
    # by Korf, Yandell and Bedell (O'Reilly, 2003)
    lower_bound = np.int32(np.round(1/ka_k, 0))
    exp_hsp_len = np.int32(np.round(log_const / ka_h, 0))

    # difference here since there can be N-vs-N
    # comparisons for SCIDDO, not only 1-vs-N
    eff_grp1_len = np.int32(m - (sm * exp_hsp_len))
    if eff_grp1_len < lower_bound:
        eff_grp1_len = lower_bound

    eff_grp2_len = np.int32(n - (sn * exp_hsp_len))
    if eff_grp2_len < lower_bound:
        eff_grp2_len = lower_bound
    return exp_hsp_len, eff_grp1_len, eff_grp2_len


def compute_scan_parameters(chromosomes, scoring_params, group1, group2,
                            baselengths, grouplengths, unit):
    """
    :param chromosomes: chromosome information
     :type: Pandas DataFrame
    :param scoring_params: scoring parameters for one scoring
     :type: Pandas DataFrame
    :param group1: group 1 information
     :type: dict
    :param group2: group 2 information
     :type: dict
    :param baselengths: base length per chromosome
     :type: dict
    :param grouplengths: total sequence length per group and chromosome or "linear"
     :type: str or Pandas DataFrame
    :param unit: unit for logarithm
     :type: str
    :return:
    """
    param_lut = dict()
    suffix = 'linear'
    for chrom in chromosomes.index:
        if chrom == 'genome':
            continue
        tmp = dict()
        tmp['binsize'] = chromosomes.at[chrom, 'binsize']
        tmp['chrom_len'] = chromosomes.at[chrom, 'bins']
        ka_k = scoring_params.at[chrom, 'ka_k']
        ka_h = scoring_params.at[chrom, 'ka_h']
        tmp['chrom'] = chrom
        tmp['ka_k'] = ka_k
        tmp['ka_h'] = ka_h
        tmp['ka_lambda'] = scoring_params.at[chrom, 'ka_lambda']
        base_length = baselengths[chrom]
        if base_length == tmp['chrom_len']:
            prefix = 'full'
        elif base_length == 0:
            prefix = 'pairwise'
        else:
            prefix = 'variable'

        if base_length > 0:
            if isinstance(grouplengths, str) and grouplengths == 'linear':
                # total number of samples in group
                sm = group1['total']
                sn = group2['total']
                comp = sn * sm
                eff_grp1 = comp * base_length
                eff_grp2 = 0
                exp_hsp = -1
                suffix = 'linear'
            elif isinstance(grouplengths, pd.DataFrame):
                # means: group size is set to adaptive
                # correct group sizes need to be computed
                # before this
                m = grouplengths.at[chrom, 'group1']
                # number of singletons plus number of replicate groups
                sm = group1['singletons'] + group1['rep_groups']
                n = grouplengths.at[chrom, 'group2']
                sn = group2['singletons'] + group2['rep_groups']
                eff_grp1 = m - base_length
                eff_grp2 = n
                exp_hsp = -1
                suffix = 'adaptive'
            else:
                raise ValueError('Unexpected type of argument groupsize: {}'.format(type(grouplengths)))
            # effective sequence length computation is removed - theory seems
            # only to be developed for alignment case
            # exp_hsp, eff_grp1, eff_grp2 = compute_expected_hsp_length(unit, ka_k, ka_h,
            #                                                           m, sm, n, sn)
        else:
            exp_hsp, eff_grp1, eff_grp2 = 0, 0, 0

        assert all(np.isfinite([eff_grp1, eff_grp2, exp_hsp, base_length])), \
            'Group, expected HSP or base length is invalid: G1 {}, G2 {}, EXP {}, BL {}'.format(eff_grp1,
                                                                                                eff_grp2,
                                                                                                exp_hsp,
                                                                                                base_length)

        tmp['exp_hsp_len'] = exp_hsp
        tmp['eff_grp1_len'] = eff_grp1
        tmp['eff_grp2_len'] = eff_grp2
        tmp['seq_base_len'] = base_length
        tmp['len_norm'] = prefix + '_' + suffix
        param_lut[chrom] = tmp
    return param_lut


def compute_expect(score, m, n, ka_k=None, ka_lambda=None, is_nat=True):
    """
    :param score:
    :param m: sequence length of group 1
    :param n: sequence length of group 2
    :param ka_k:
    :param ka_lambda:
    :param is_nat: scores are nat scores, not raw scores
    :return:
    """
    np.seterr(all='raise')
    L = m + n
    if hasattr(score, '__iter__'):
        if isinstance(score, np.ndarray):
            # can use vectorized operations
            if is_nat:
                res = np.array(np.exp(-1 * score) * L, dtype=score.dtype)
            else:
                res = np.array(ka_k * L * np.exp(-1 * ka_lambda * score), dtype=score.dtype)
        elif isinstance(score, pd.Series):
            if is_nat:
                res = pd.Series(np.exp(-1 * score) * L, dtype=score.dtype)
            else:
                res = pd.Series(ka_k * L * np.exp(-1 * ka_lambda * score), dtype=score.dtype)
        else:
            # "compatibility" option
            # could be Python list or something else...
            if is_nat:
                res = list(map(lambda s: L * np.exp(-1 * s), score))
            else:
                res = list(map(lambda s: ka_k * L * np.exp(-1 * ka_lambda * s), score))
    else:
        # for single score value
        if is_nat:
            try:
                res = np.exp(-1 * score) * L
            except FloatingPointError as fpe:
                # numerical underflow happens 3x for test data set because
                # raw / unmerged segments are used; catch that here
                res = np.exp(-1 * min(700, score)) * L
                #raise FloatingPointError('FPE - {}: score {} / L {}'.format(str(fpe), score, L))
        else:
            res = ka_k * L * np.exp(-1 * ka_lambda * score)
    return res


def single_segment_score_pv(score, raw=True):
    """
    Compute p-value for normalized local score of
    a single high scoring segment.

    Computes formula [1] in Karlin & Altschul, PNAS 1993

    Prob(S' >= x) ~ 1 - exp(-exp(-x))

    :param score:
    :param raw: return raw P-value instead of -log10(pv)
    :return:
    """
    with mpm.workprec(NUM_PREC_KA_PV):
        x = mpm.convert(score)
        complement = mpm.convert('1')
        exponent = mpm.fneg(mpm.exp(mpm.fneg(x)))
        res = mpm.fsub(complement, mpm.exp(exponent))
        if not raw:
            res = mpm.fneg(mpm.log10(res))
            res = float(res)

    # Equivalent implementation using Python standard library:
    #
    # x = score
    # res = 1 - math.exp(-math.exp(-x))
    # if not raw:
    #     res = -1 * math.log10(res)
    return res


def single_segment_expect_pv(expect, raw=True):
    """
    Compute p-value for an expect value of
    a single high scoring segment.

    Prob(E >= x) ~ 1 - exp(-E)

    This function is equivalent to
    single_segment_score_pv
    as long as identical units (log)
    are used to compute scores and expect

    :param expect:
    :param raw: return raw P-value instead of -log10(pv)
    :return:
    """
    with mpm.workprec(NUM_PREC_KA_PV):
        x = mpm.convert(expect)
        complement = mpm.convert('1')
        res = mpm.fsub(complement, mpm.exp(mpm.fneg(x)))
        if not raw:
            res = mpm.fneg(mpm.log10(res))
            res = float(res)

    return res


def multi_segment_score_pv(score, num_segments, raw=True):
    """
    Compute p-value for normalized score when considering
    multiple high scoring segments. This functions considers
    the normalized score Sr', i.e., the normalized score
    of the HSP at rank r. For r=1, this formula is equivalent
    to single_segment_score_pv

    Computes formula [3] in Karlin & Altschul, PNAS 1993

    Prob(Sr' >= x) ~ 1 - exp(-exp(-x)) * SUM (k=0 ... r - 1) { exp(-kx) / k! }

    Implementation detail:
    Python's range is not right-inclusive,
    go up to r, not r - 1 for summation

    :param score:
    :param num_segments:
    :param raw: return raw P-value instead of -log10(pv)
    :return:
    """

    with mpm.workprec(NUM_PREC_KA_PV):
        def create_summand(sum_x, k):
            prec_k = mpm.convert(k)
            enum = mpm.exp(mpm.fneg(mpm.fmul(prec_k, sum_x)))
            denom = mpm.factorial(prec_k)
            summand = mpm.fdiv(enum, denom)
            return summand
        x = mpm.convert(score)
        r = num_segments
        complement = mpm.convert('1')
        factor1 = mpm.exp(mpm.fneg(mpm.exp(mpm.fneg(x))))
        factor2 = mpm.fsum(map(lambda k: create_summand(x, k), range(0, r)))
        res = mpm.fsub(complement, mpm.fmul(factor1, factor2))
        if not raw:
            res = mpm.fneg(mpm.log10(res))
            res = float(res)

    # Equivalent implementation using Python standard library:
    #
    # x = score
    # r = num_segments
    # factor_1 = math.exp(-math.exp(-x))
    # factor_2 = math.fsum(map(lambda k: math.exp(-k * x) / math.factorial(k), range(0, r)))
    # res = 1 - factor_1 * factor_2
    # if not raw:
    #     res = -1 * math.log10(res)
    return res


def summed_score_pv(sum_score, num_segments, raw=True):
    """
    Compute p-value over sum of scores for top r segments.
    As opposed to multi_segment_pv, this considers the
    total sum of scores for the top r scoring segments
    together.

    Computes formula [5] in Karlin & Altschul, PNAS 1993

    Prob(Tr >= x) ~ exp(-x) * x**(r-1) / r! * (r - 1)!

    where Tr = S1' + ... + Sr' is the sum over the
    normalized scores of the top r segments

    :param sum_score:
    :param num_segments:
    :param raw: return raw P-value instead of -log10(pv)
    :return:
    """
    with mpm.workprec(NUM_PREC_KA_PV):
        x = mpm.convert(sum_score)
        r = mpm.convert(str(num_segments))
        rm1 = mpm.fsub(r, mpm.convert('1'))
        enum = mpm.fmul(mpm.exp(mpm.fneg(x)), mpm.power(x, rm1))
        denom = mpm.fmul(mpm.factorial(r), mpm.factorial(rm1))
        res = mpm.fdiv(enum, denom)
        if not raw:
            res = mpm.fneg(mpm.log10(res))
            res = float(res)

    # Equivalent implementation using Python standard library:
    #
    # r = num_segments
    # x = sum_score
    # enum = math.exp(-x) * math.pow(x, (r - 1))
    # denom = math.factorial(r) * math.factorial(r - 1)
    # res = enum / denom
    # if not raw:
    #     res = -1 * math.log10(res)
    return res


def compute_lambda_equation(ka_lambda, score_values, score_probs):
    """
    This computes equation [4] in Karlin & Altschul, PNAS 1990
    (minus 1 to equal 0 for root finding).
    This can be used to estimate the value of the
    Karlin-Altschul parameter lambda (see estimate_ka_lambda)

    :param ka_lambda:
    :param score_values:
    :param score_probs:
    :return:
    """
    res = (sum(p * np.exp(ka_lambda * s) for s, p in zip(score_values, score_probs)))
    return res - 1


def estimate_ka_lambda(score_values, score_probs, init=0.5):
    """
    Initial guess follows convention from BLAST. Empirically,
    lambda is expected to be somewhere in the interval (0, 2)

    :param score_values:
    :param score_probs:
    :param init:
    :return:
    """
    lambda_est = opt.newton(compute_lambda_equation, init,
                            args=(score_values, score_probs),
                            tol=10e-6, maxiter=100)
    return lambda_est


def compute_ka_h(ka_lambda, score_values, score_probs):
    """
    This computes equation [5] in Altschul, J. Mol. Biol. 1991
    (expressed in base frequencies). The result
    is the Karlin-Altschul parameter H (the entropy
    of the scoring matrix).
    This formulation can also be found in the BLAST
    developer notes - you can find a copy here:
    github/ptrebert/sciddo/misc/blast_stats/Gertz_2005_BLAST_scoing.pdf


    :param ka_lambda:
    :param score_values:
    :param score_probs:
    :return:
    """
    ka_h = ka_lambda * (sum(s * p * np.exp(s*ka_lambda) for s, p in zip(score_values, score_probs)))
    return ka_h


def _derive_score_boundaries(low, high):
    """
    Utility function to derive lower and upper
    boundaries for the actually observed values

    This exists just to accommodate the BLAST C routines

    :param low:
    :param high:
    :return:
    """
    lowest_min = np.iinfo(eval(NUM_MAX_SIZE_SCORE)).min
    highest_max = np.iinfo(eval(NUM_MAX_SIZE_SCORE)).max
    assert lowest_min < low < high < highest_max,\
        'Observed scores out of allowed range {} ... {}: {} - {}'.format(lowest_min, highest_max, low, high)
    lower_bound = 0
    upper_bound = 0
    for x in range(1, int(np.log2(highest_max + 1))):
        v = 2**x
        if -v <= low and lower_bound == 0:
            lower_bound = -v
        if v >= high and upper_bound == 0:
            upper_bound = v
        if lower_bound != 0 and upper_bound != 0:
            break
    assert lower_bound < 0 < upper_bound, \
        'Could not derive score boundaries: {} ... {}'.format(lower_bound, upper_bound)
    return lower_bound, upper_bound


def init_karlin_altschul_estimation():
    """
    :return:
    """
    try:
        import karlin
        raise RuntimeError('karlin module can be imported. Please run Karlin-Altschul'
                           ' parameter estimation via the karlin module function'
                           ' "estimateParameters".')
    except ImportError:
        sys.stderr.write('\nscidlib.statistics.init_karlin_altschul_estimation:\n'
                         ' Could not import karlin module - running Karlin-Altschul\n'
                         ' parameter estimation via the ctypes wrapper interface\n'
                         ' is discouraged and likely to fail.\n')
    so_abs_path = os.path.abspath(os.path.dirname(__file__))
    so_abs_fullpath = os.path.join(so_abs_path, 'karlin.so')
    assert os.path.isfile(so_abs_fullpath), \
        'Could not find shared library object karlin.so under path {}'.format(so_abs_fullpath)
    _karlin = cty.CDLL(so_abs_fullpath)
    # int8_t
    # ComputeKarlinAltschulParameters(
    # double* scorefreq, /* Score frequencies, must sum to 1 */
    # int16_t obs_min, /* minimum observed score */
    # int16_t obs_max, /* maximum observed score */
    # int16_t score_min, /* minimum boundary on scores (not minimum observed) */
    # int16_t score_max, /* maximum boundary on scores (not maximum observed) */
    # double* lambda, /* pointer to parameter lambda */
    # double* entropy, /* pointer to parameter H, entropy */
    # double* k /* pointer to parameter K */)

    _karlin.ComputeKarlinAltschulParameters.argtypes = (cty.POINTER(cty.c_double),  # score frequencies
                                                        cty.c_int16,  # min observed score
                                                        cty.c_int16,  # max observed score
                                                        cty.c_int16,  # min score boundary
                                                        cty.c_int16,  # max score boundary
                                                        cty.POINTER(cty.c_double),  # store lambda
                                                        cty.POINTER(cty.c_double),  # store H
                                                        cty.POINTER(cty.c_double))  # store K

    def estimate_karlin_altschul_parameters(lowest, highest, score_probs):
        """
        Estimate Karlin-Altschul parameters, uses BLAST C routines

        :param lowest: lowest observed score
        :param highest: highest observed score
        :param score_probs: iterable of score probabilities (can include scores with 0 probability)
        :return: estimates for lambda, H/entropy and K
        """
        lower_bound, upper_bound = _derive_score_boundaries(lowest, highest)

        # build C pointer type
        multi_array = cty.c_double * len(score_probs)
        single_array = cty.c_double * 1

        prob_array = multi_array(*score_probs)
        lambda_est = single_array(0.)
        h_est = single_array(0.)
        k_est = single_array(0.)

        res_int = _karlin.ComputeKarlinAltschulParameters(prob_array,
                                                          cty.c_int16(lowest),
                                                          cty.c_int16(highest),
                                                          cty.c_int16(lower_bound),
                                                          cty.c_int16(upper_bound),
                                                          lambda_est,
                                                          h_est,
                                                          k_est)
        # res int should always be zero,
        # no need to check here
        result_lambda = float(lambda_est[0])
        result_h = float(h_est[0])
        result_k = float(k_est[0])

        assert all([v > 0 for v in [result_lambda, result_h, result_k]]), \
            'Karlin-Altschul parameter estimation failed:' \
            ' lambda {} / H {} / K {}'.format(result_lambda, result_h, result_k)
        return result_lambda, result_h, result_k
    return estimate_karlin_altschul_parameters
