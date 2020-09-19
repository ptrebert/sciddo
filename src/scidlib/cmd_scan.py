# coding=utf-8

import os as os
import collections as col
import itertools as itt
import multiprocessing as mp

import numpy as np
import numpy.random as rng
import pandas as pd
import intervaltree as ivt

from scidlib import ROOT_SCORE, PATH_MD_CHROM, ROOT_SCAN, \
    PATH_MD_DESIGN, ROOT_HSP, PATH_MD_COMPARISON, \
    PATH_MD_COMP_PARAMS
from scidlib.sample_handling import get_replicate_pairs, derive_merge_labels
from scidlib.algorithms import get_all_max_scoring_subseq
from scidlib.statistics import compute_scan_parameters, normalize_scores, \
    single_segment_expect_pv, compute_expect, compute_expected_hsp_length


def add_scan_cmd_parser(subparsers):
    """
    :param subparsers:
    :return:
    """
    parser_scan = subparsers.add_parser('scan',
                                        help='Scan dataset for differential chromatin domains')

    grp = parser_scan.add_argument_group('Input/output path')
    grp.add_argument('--sciddo-data', '-d', type=str, dest='dataset', required=True,
                     help='Path to input SCIDDO dataset in HDF5 format. Computed'
                          'scoring matrices will be added to the dataset,'
                          'there is no separate output file.')
    grp.add_argument('--run-out', '-ro', type=str, required=True, dest='runout',
                     help='Specify the full path to the output file.'
                          ' All raw results will be stored in this file under'
                          ' the path {} and merged, final segments under'
                          ' the path {}'.format(os.path.join(ROOT_SCAN, '<scoring>', 'chrNN'),
                                                os.path.join(ROOT_HSP, '<scoring>', 'chrNN')))

    grp = parser_scan.add_argument_group('Scan parameters')
    grp.add_argument('--scoring', '-sc', type=str, nargs='+', required=True, dest='scoring',
                     help='Specify the name of the scoring matrix/matrices to be used.')
    grp.add_argument('--count-length', '-cl', type=str, choices=['full', 'variable'],
                     dest='countlength', default='full',
                     help='Count the length of the sequence per sample either in "full"'
                          ' (= total number of bins) or just as "variable" bins'
                          ' (= bins that differ at least once across the entire'
                          ' dataset). Note that the option "variable" should be'
                          ' considered experimental.'
                          ' DEFAULT: full')
    grp.add_argument('--adjust-group-length', '-agl', type=str, choices=['linear', 'adaptive'],
                     dest='grouplength', default='adaptive',
                     help='When comparing groups of samples, adjust the combined sequence'
                          ' length per group either in a "linear" fashion (= add complete'
                          ' chromosome length for each additional comparison'
                          ' depending on the option set for "--count-length")'
                          ' or in an "adaptive" fashion (= the first sample is counted'
                          ' according the option set for "--count-length",'
                          ' the length of all additional samples is counted as number of bins'
                          ' that differ between this sample and all other samples in the group.'
                          ' Note that the option "adaptive" should only be chosen for homogeneous'
                          ' groups of samples such as biological replicates.'
                          ' DEFAULT: adaptive')
    grp.add_argument('--run-baseline', '-rbl', type=str, choices=['replicate', 'random'],
                     dest='baseline', default='',
                     help='Run baseline comparisons applying one of the following'
                          ' strategies:'
                          ''
                          '"replicate": compare all replicates in a 1-vs-1 fashion.'
                          ' This baseline run uses standard "full" and "variable"'
                          ' length counting plus the "pairwise variable" (set length'
                          ' of both sequences to the number of bins different only'
                          ' between the two replicates being compared).'
                          ''
                          '"random": randomize state segmentation maps before'
                          ' scanning for maximal scoring segments. Note that this kind'
                          ' of scan run continues until a specified number of segments'
                          ' has been found (which might take some time). See parameter'
                          ' "--num-random" below')
    grp.add_argument('--num-random', '-rand', type=int, dest='numrandom', default=1000,
                     help='Specify number of maximal scoring segments to be identified'
                          ' before the baseline run with strategy "random" is stopped.'
                          ' Default: 1000')
    grp.add_argument('--compute-raw-stats', '-rs', type=int, default=-1, dest='rawstats',
                     help='Compute statistics for the top N high scoring segment'
                          ' pairs (unmerged segments). A value of -1 disables'
                          ' statistical computation, a value of 0 computes statistics'
                          ' for all high scoring segments. DEFAULT: -1')
    grp.add_argument('--merge-segments', '-mrg', action='store_true', dest='merge', default=False,
                     help='Merge overlapping HSPs from individual comparisons.'
                          ' This option should only be set when two homogeneous'
                          ' groups of samples (e.g., two groups of replicates)'
                          ' are being compared.'
                          ' DEFAULT: False')
    grp.add_argument('--compute-merged-stats', '-ms', type=int, default=0, dest='mergedstats',
                     help='Compute statistics for the top N high scoring segment'
                          ' pairs (after merging overlapping segments). A value'
                          ' of -1 disables computation of statistics, a value of 0'
                          ' computes statistics for all high scoring segments after'
                          ' merging. DEFAULT: 0')
    grp.add_argument('--bootstraps', '-bs', type=int, default=1000, dest='bootstraps',
                     help='Specify number of bootstraps to estimate segment mean score when'
                          ' merging overlapping segments. Set to 0 for no bootstrapping.'
                          ' DEFAULT: 1000')

    mex_grp = grp.add_mutually_exclusive_group(required=False)
    mex_grp.add_argument('--samples', '-smp', type=str, nargs='*', default=[], dest='samples',
                         help='Restrict the scan process to these samples given as space-separated'
                              ' list of sample labels. If left empty and "--select-groups" is not set,'
                              ' the default behavior is to run an all-vs-all scan using all'
                              ' samples in the dataset.'
                              ' DEFAULT: <empty>')
    mex_grp.add_argument('--select-groups', '-grp', action='store_true', default=False, dest='selectgroups',
                         help='Set this option to enable selecting groups of samples, either by'
                              ' sample name or by their properties as indicated in the design matrix.'
                              ' DEFAULT: False')

    grp = parser_scan.add_argument_group('Specify sample groups',
                                         description='If the option "--select-groups" is set,'
                                                     ' you need to specify the sample names or '
                                                     ' properties as given in the design'
                                                     ' matrix to select the respective groups.')
    grp.add_argument('--group1', '-grp1', type=str, nargs='*', default=[], dest='group1',
                     help='Specify names/property to select samples by (group 1).'
                          ' DEFAULT: <empty>')
    grp.add_argument('--group2', '-grp2', type=str, nargs='*', default=[], dest='group2',
                     help='Specify names/property to select samples by (group 2).'
                          ' DEFAULT: <empty>')

    parser_scan.set_defaults(execute=_run_cmd_scan)
    return subparsers


def __merge_interval_data(reduced_data, new_item):
    """
    :param reduced_data:
    :param new_item:
    :return:
    """
    try:
        reduced_data.append(new_item)
        return reduced_data
    except AttributeError:
        return [reduced_data, new_item]


def merge_raw_segments(segments, bootstrap=0):
    """
    :param segments:
    :param bootstrap:
    :return:
    """
    # note: intervaltree operates on half-open intervals,
    # hence, there is a +1 for the end bin
    ivtree = ivt.IntervalTree.from_tuples([s, e + 1, (r, n)] for s, e, r, n in zip(segments['start_bin'],
                                                                                   segments['end_bin'],
                                                                                   segments['raw_score'],
                                                                                   segments['nat_score']))
    ivtree.merge_overlaps(__merge_interval_data)
    # when preparing the final segments
    # subtract 1 again from end (start - end inclusive)
    # final fields:
    # start_bin, end_bin, raw_score, norm_score, norm_score_std, num_mrg_segments, bootstraps
    final = []
    for iv in ivtree.items():
        # if a segment has no overlaps
        # its data is never appended
        # to a list
        if isinstance(iv.data, list):
            raw_score_sample = [t[0] for t in iv.data]
            norm_score_sample = [t[1] for t in iv.data]
            if bootstrap > 0:
                norm_sampling_dist = rng.choice(norm_score_sample, size=(bootstrap, len(iv.data)), replace=True)
                norm_sampling_means = norm_sampling_dist.mean(axis=1)
                norm_est_mean = norm_sampling_means.mean()
                norm_sampling_std = norm_sampling_means.std()
                
                raw_sampling_dist = rng.choice(raw_score_sample, size=(bootstrap, len(iv.data)), replace=True)
                raw_sampling_means = raw_sampling_dist.mean(axis=1)
                raw_est_mean = np.round(raw_sampling_means.mean(), 0)
                raw_sampling_std = raw_sampling_means.std()
            else:
                norm_est_mean = np.mean(norm_score_sample)
                norm_sampling_std = 0

                raw_est_mean = np.round(np.mean(raw_score_sample), 0)
                raw_sampling_std = 0

            segment = iv.begin, iv.end - 1, raw_est_mean, raw_sampling_std, norm_est_mean, norm_sampling_std, len(iv.data), bootstrap
        elif isinstance(iv.data, tuple):
            segment = iv.begin, iv.end - 1, iv.data[0], 0, iv.data[1], 0, 1, bootstrap
        else:
            raise ValueError('Unexpected interval data: {} ({})'.format(iv.data, type(iv.data)))
        final.append(segment)
    final = sorted(final, key=lambda x: x[3], reverse=False)

    merged_segments = pd.DataFrame(final, index=np.arange(len(final)),
                                   columns=['start_bin', 'end_bin',
                                            'raw_score', 'raw_score_std',
                                            'nat_score', 'nat_score_std',
                                            'num_merged', 'num_bootstraps'])
    return merged_segments


def add_segment_coordinates(segments, binsize):
    """
    :param segments:
    :param binsize:
    :return:
    """
    segments['start_bp'] = segments['start_bin'] * binsize
    segments['end_bp'] = segments['end_bin'] * binsize
    segments['num_bins'] = segments['end_bin'] - segments['start_bin']
    # segments are stored as start-end inclusive, adjust number of bins
    segments['num_bins'] += 1
    return segments


def find_state_hsp(params):
    """
    :param params:
    :return:
    """
    # args.dataset, s1, s2, c, s, args.rawstats, ka_scan_params, rep
    fpath, s1, s2, chrom, scoring_name, rawstats, ka_params, is_rep = params
    with pd.HDFStore(fpath, 'r') as hdf:
        s1_states = hdf[os.path.join('state', s1, chrom)]
        s2_states = hdf[os.path.join('state', s2, chrom)]
        scoring = hdf[os.path.join(ROOT_SCORE, scoring_name, 'matrix')]
        scores = scoring.lookup(s1_states, s2_states)
        segments = pd.DataFrame.from_records(get_all_max_scoring_subseq(scores),
                                             columns=['raw_score', 'start_bin', 'end_bin'])

        if ka_params['seq_base_len'] == 0:
            # need to derive effective lengths
            # based on pairwise difference
            diff_bins = np.array(s1_states != s2_states, dtype=np.int8).sum()
            ka_k = ka_params['ka_k']
            ka_h = ka_params['ka_h']
            exp_hsp, eff_grp1, eff_grp2 = compute_expected_hsp_length('e', ka_k, ka_h, diff_bins, 1, diff_bins, 1)
            ka_params['exp_hsp_len'] = exp_hsp
            ka_params['eff_grp1_len'] = eff_grp1
            ka_params['eff_grp2_len'] = eff_grp2
            ka_params['seq_base_len'] = diff_bins

    # no HSP identified
    if segments.empty or segments is None:
        return chrom, None, scoring_name

    # if there was a raw score <= 0, this should point
    # to something being wrong in the Ruzzo & Tompa implementation
    assert segments['raw_score'].min() > 0, 'Minimal raw HSP score is <= 0'

    segments = add_segment_coordinates(segments, ka_params['binsize'])

    segments['nat_score'] = normalize_scores(segments['raw_score'],
                                             ka_params['ka_lambda'],
                                             ka_params['ka_k'],
                                             'e', m=1, n=1)
    # to make scores comparable across chromosomes, need to compute
    # the length corrected version (which is otherwise done as part
    # of the Expect computation) - this is only relevant for cases
    # where the Expect is not used as "quality measure"
    segments['nat_score_lnorm'] = normalize_scores(segments['raw_score'],
                                                   ka_params['ka_lambda'],
                                                   ka_params['ka_k'], units='e',
                                                   m=ka_params['eff_grp1_len'],
                                                   n=ka_params['eff_grp2_len'])
    segments = compute_ka_statistics(segments, rawstats, ka_params)
    segments.sort_values(by='segment_expect', ascending=False,
                         axis=0, inplace=True)
    segments.reset_index(drop=True, inplace=True)
    segments['sample1'] = s1
    segments['sample2'] = s2
    segments['len_norm'] = ka_params['len_norm']
    dtypes = {'start_bin': np.int32, 'end_bin': np.int32, 'num_bins': np.int32,
              'start_bp': np.int32, 'end_bp': np.int32,
              'raw_score': np.int32, 'nat_score': np.float64, 'nat_score_lnorm': np.float64,
              'segment_pv': np.float64, 'segment_expect': np.float64,
              'sample1': str, 'sample2': str, 'len_norm': str}
    segments = segments.astype(dtypes)
    return chrom, segments, scoring_name


def find_random_hsp(params):
    """
    :param params:
    :return:
    """
    # args.dataset, s1, s2, c, s, args.rawstats, ka_scan_params, rep
    fpath, s1, s2, chrom, scoring_name, rawstats, ka_params, is_rep = params
    segments = None
    with pd.HDFStore(fpath, 'r') as hdf:
        s1_states = hdf[os.path.join('state', s1, chrom)].values
        s2_states = hdf[os.path.join('state', s2, chrom)].values
        scoring = hdf[os.path.join(ROOT_SCORE, scoring_name, 'matrix')]
        for _ in range(10):
            scores = scoring.lookup(rng.permutation(s1_states),
                                    rng.permutation(s2_states))

            segments = pd.DataFrame.from_records(get_all_max_scoring_subseq(scores),
                                                 columns=['raw_score', 'start_bin', 'end_bin'])
            if segments.empty:
                continue
            else:
                segments = segments.loc[segments['raw_score'] == segments['raw_score'].max(), :].copy()
                if segments.shape[0] > 1:
                    # randomly pick just one entry
                    segments = segments.iloc[[0], :].copy()
                break
    # no HSP identified
    if segments.empty or segments is None:
        return chrom, None, scoring_name

    # if there was a raw score <= 0, this should point
    # to something being wrong in the Ruzzo & Tompa implementation
    assert segments['raw_score'].min() > 0, 'Minimal raw HSP score is <= 0'

    segments = add_segment_coordinates(segments, ka_params['binsize'])

    segments['nat_score'] = normalize_scores(segments['raw_score'],
                                             ka_params['ka_lambda'],
                                             ka_params['ka_k'],
                                             'e', m=1, n=1)
    # to make scores comparable across chromosomes, need to compute
    # the length corrected version (which is otherwise done as part
    # of the Expect computation) - this is only relevant for cases
    # where the Expect is not used as "quality measure"
    segments['nat_score_lnorm'] = normalize_scores(segments['raw_score'],
                                                   ka_params['ka_lambda'],
                                                   ka_params['ka_k'], units='e',
                                                   m=ka_params['eff_grp1_len'],
                                                   n=ka_params['eff_grp2_len'])
    segments = compute_ka_statistics(segments, rawstats, ka_params)
    segments.sort_values(by='segment_expect', ascending=False,
                         axis=0, inplace=True)
    segments.reset_index(drop=True, inplace=True)
    segments['sample1'] = s1
    segments['sample2'] = s2
    segments['len_norm'] = ka_params['len_norm']
    dtypes = {'start_bin': np.int32, 'end_bin': np.int32, 'num_bins': np.int32,
              'start_bp': np.int32, 'end_bp': np.int32,
              'raw_score': np.int32, 'nat_score': np.float64, 'nat_score_lnorm': np.float64,
              'segment_pv': np.float64, 'segment_expect': np.float64,
              'sample1': str, 'sample2': str, 'len_norm': str}
    segments = segments.astype(dtypes)
    return chrom, segments, scoring_name


def merge_state_hsp(params):
    """
    :param params:
    :return:
    """
    data_path, segments, bootstraps, ka_params, label1, label2, mergedstats, num_comp = params
    _, _, scoring, chrom = data_path.split('/')

    merged_segments = merge_raw_segments(segments, bootstraps)
    merged_segments['group1'] = label1
    merged_segments['group2'] = label2
    merged_segments['num_comparisons'] = num_comp
    merged_segments = add_segment_coordinates(merged_segments, ka_params['binsize'])
    merged_segments = compute_ka_statistics(merged_segments, mergedstats, ka_params)
    merged_segments['nat_score_lnorm'] = normalize_scores(merged_segments['raw_score'],
                                                          ka_params['ka_lambda'],
                                                          ka_params['ka_k'], units='e',
                                                          m=ka_params['eff_grp1_len'],
                                                          n=ka_params['eff_grp2_len'])
    merged_segments.sort_values(by='segment_expect', ascending=False,
                                axis=0, inplace=True)
    merged_segments.reset_index(drop=True, inplace=True)
    dtypes = {'start_bin': np.int32, 'end_bin': np.int32, 'num_bins': np.int32,
              'start_bp': np.int32, 'end_bp': np.int32,
              'raw_score': np.int32, 'raw_score_std': np.float32,
              'nat_score': np.float64, 'nat_score_std': np.float64,  'nat_score_lnorm': np.float64,
              'num_merged': np.int32, 'num_comparisons': np.int32, 'num_bootstraps': np.int32,
              'segment_pv': np.float64, 'segment_expect': np.float64, 'group1': str, 'group2': str}
    merged_segments = merged_segments.astype(dtypes)
    return chrom, scoring, merged_segments


def compute_ka_statistics(segment_scores, threshold, ka_params):
    """
    :param segment_scores:
    :param threshold:
    :param ka_params:
    :return:
    """
    segment_scores.sort_values(by='nat_score', axis=0, ascending=False, inplace=True)
    segment_scores.reset_index(drop=True, inplace=True)
    segment_pv = np.ones(segment_scores.shape[0], dtype=np.float64)
    segment_pv *= -1
    segment_expect = np.ones(segment_scores.shape[0], dtype=np.float64)
    segment_expect *= -1
    skip = False
    if threshold == 0:
        threshold = segment_scores.shape[0]
    elif threshold < 0:
        skip = True
    else:
        pass
    if not skip:
        np.seterr(all='raise')
        m, n = ka_params['eff_grp1_len'], ka_params['eff_grp2_len']
        for idx, score in enumerate(segment_scores['nat_score']):
            try:
                seg_exp = compute_expect(score, m, n, is_nat=True)
                segment_expect[idx] = seg_exp
            except OverflowError:
                segment_pv[idx] = -2
                seg_exp = None
            except ValueError:  # math domain error
                segment_pv[idx] = -3
                seg_exp = None
            except TypeError:
                segment_pv[idx] = -4
                seg_exp = None

            if seg_exp is not None:
                try:
                    seg_pv = single_segment_expect_pv(seg_exp, False)
                    segment_pv[idx] = seg_pv
                except OverflowError:
                    segment_pv[idx] = -2
                except ValueError:  # math domain error
                    segment_pv[idx] = -3
                except TypeError:
                    segment_pv[idx] = -4
            else:
                segment_pv[idx] = -5

            if idx + 1 >= threshold:
                break
    segment_scores['segment_pv'] = segment_pv
    segment_scores['segment_expect'] = segment_expect
    return segment_scores


def get_adaptive_group_size(param):
    """
    :param param:
    :return:
    """
    fpath, chrom, baselen, group, samples = param
    # group_size = 0
    # group_states = None
    with pd.HDFStore(fpath, 'r') as hdf:
        state_data = [hdf[os.path.join('state', s, chrom)] for s in samples]
        group_size = compute_adaptive_group_size(state_data, baselen)
        # for s in samples:
        #     data_path = os.path.join('state', s, chrom)
        #     data = hdf[data_path]
        #     if group_size == 0:
        #         group_size += baselen
        #         group_states = pd.DataFrame([data])
        #     else:
        #         diff_bin = (data != group_states).any(axis=0).sum()
        #         group_states = group_states.append(data, ignore_index=True)
        #         group_size += diff_bin
    return chrom, group, group_size


def compute_adaptive_group_size(samples, base_length):
    """
    :param samples:
    :param base_length:
    :return:
    """
    # Note to self: could be implemented with pandas.DataFrame.nunique,
    # but for some reason, that is super slow (~4x slower)
    group_size = 0
    for s in samples:
        if group_size == 0:
            group_size += base_length
            group_states = pd.DataFrame([s])
        else:
            diff_bin = (s != group_states).any(axis=0).sum()
            group_states = group_states.append(s, ignore_index=True)
            group_size += diff_bin
    return group_size

  
def collect_adaptive_group_size(args, base_length, group1, group2):
    """
    :param args:
    :param group1:
    :param group2:
    :return:
    """
    with pd.HDFStore(args.dataset, 'r') as hdf:
        chromosomes = hdf[PATH_MD_CHROM].index.tolist()

    groupsizes = pd.DataFrame(np.zeros((len(chromosomes), 2), dtype=np.int64),
                              columns=['group1', 'group2'], index=sorted(chromosomes))
    params = []
    for c in chromosomes:
        params.append((args.dataset, c, base_length[c], 'group1', group1['samples']))
        params.append((args.dataset, c, base_length[c], 'group2', group2['samples']))

    with mp.Pool(args.workers) as pool:
        resit = pool.imap_unordered(get_adaptive_group_size, params)
        for chrom, group, groupsize in resit:
            groupsizes.loc[chrom, group] = groupsize

    return groupsizes


def collect_base_lengths(fpath, strategy):
    """
    :param fpath:
    :param strategy:
    :return:
    """
    with pd.HDFStore(fpath, 'r') as hdf:
        if strategy == 'full':
            md_chrom = hdf[PATH_MD_CHROM]
            baselengths = {row.Index: int(row.bins) for row in md_chrom.itertuples()}
        elif strategy == 'variable':
            # variable length is stored as
            # part of the scoring parameters,
            # but identical for all scorings
            score_params = [k for k in hdf.keys() if k.startswith('/scoring') and k.endswith('/parameters')]
            md_scoring = hdf[score_params[0]]
            baselengths = {row.Index: int(row.var_len) for row in md_scoring.itertuples()}
        else:
            baselengths = None
            raise ValueError('Unexpected value for count length: {}'.format(strategy))
    return baselengths


def count_replicate_groups(samples, rep_pairs):
    """
    :param samples:
    :param rep_pairs:
    :return:
    """
    rep_groups = col.defaultdict(set)
    used_pairs = [r for r in rep_pairs if r[0] in samples and r[1] in samples]
    rep_group_id = 0
    singletons = 0
    for s in samples:
        subset = [p for p in used_pairs if s in p]
        subset = set([t[0] for t in subset]).union(set([t[1] for t in subset]))
        if not subset:
            # for this selection of samples,
            # this sample is not replicated
            singletons += 1
            continue
        rep_num = -1
        for k, v in rep_groups.items():
            if s in v:
                rep_num = k
                v.update(subset)
                break
        if rep_num != -1:
            continue
        else:
            rep_group_id += 1
            rep_groups[rep_group_id] = subset
    assert rep_group_id == len(rep_groups), \
        'Error counting replicate groups: {} vs {}'.format(rep_group_id, rep_groups)
    return rep_group_id, singletons


def build_sample_pairings(samples1, samples2, replicate_pairs, baseline=''):
    """
    :param samples1:
    :param samples2:
    :param replicate_pairs:
    :param baseline:
    :return:
    """
    # build cartesian product
    pairings = []
    # since directionality does not matter
    # need to keep track of seen combinations
    seen = set()
    group1_rep = set()
    group2_rep = set()
    for s1, s2 in itt.product(samples1, samples2):
        # this check needed
        # for case all-vs-all
        if s1 == s2:
            continue
        if (s1, s2) in seen or (s2, s1) in seen:
            continue
        is_rep = (s1, s2) in replicate_pairs or (s2, s1) in replicate_pairs
        if baseline == 'replicate' and not is_rep:
            continue
        if baseline == 'random' and is_rep:
            continue
        if is_rep:
            group1_rep.add(s1)
            group2_rep.add(s2)
        pairings.append((s1, s2, int(is_rep)))
        seen.add((s1, s2))
        seen.add((s2, s1))

    samples1 = sorted(set(p[0] for p in pairings))
    samples2 = sorted(set(p[1] for p in pairings))

    rep_groups1, sng_group1 = count_replicate_groups(samples1, replicate_pairs)
    rep_groups2, sng_group2 = count_replicate_groups(samples2, replicate_pairs)

    assert sng_group1 + rep_groups1 <= len(samples1), \
        'Group 1 composition wrong: {} / {} / {}'.format(sng_group1, rep_groups1, len(samples1))

    assert sng_group2 + rep_groups2 <= len(samples2), \
        'Group 2 composition wrong: {} / {} / {}'.format(sng_group2, rep_groups2, len(samples2))

    if baseline == 'replicate' or baseline == 'random':
        sng_per_group = {'group1': 1, 'group2': 1}
        rep_groups = {'group1': 0, 'group2': 0}
    else:
        sng_per_group = {'group1': sng_group1, 'group2': sng_group2}
        rep_groups = {'group1': rep_groups1, 'group2': rep_groups2}

    return pairings, sng_per_group, rep_groups


def build_sample_group(select_group, stored_samples, design_matrix):
    """
    :param select_group:
    :param stored_samples:
    :param design_matrix:
    :return:
    """
    if all(s in stored_samples for s in select_group):
        # user specified sample names
        samples = select_group
    else:
        # user specified sample properties
        assert design_matrix is not None, \
            'No design matrix in dataset, but group selection' \
            ' does not consist of all sample names.' \
            ' Cannot build sample group using this: {}'.format(select_group)
        samples = design_matrix.loc[design_matrix[select_group].all(axis=1), :].index.tolist()
    return sorted(set(samples))


def determine_sample_pairings(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    with pd.HDFStore(args.dataset, 'r') as hdf:
        try:
            design_matrix = hdf[PATH_MD_DESIGN]
            rep_pairs = get_replicate_pairs(design_matrix)
            rep_samples = set(p[0] for p in rep_pairs).union(set(p[1] for p in rep_pairs))
            stored_samples = set(design_matrix.index)
        except KeyError:
            logger.debug('No design matrix in dataset - assume no replicates')
            design_matrix = None
            rep_pairs = []
            rep_samples = []
            stored_samples = set([k.split('/')[2] for k in hdf.keys() if k.startswith('/state')])

    logger.debug('Identified a total of {} samples in dataset'.format(len(stored_samples)))

    # depending on normalization strategy,
    # need the following information:
    # - total number of samples per group
    # - number of singletons per group
    # - number of replicate groups per group
    group1, group2 = dict(), dict()
    if args.baseline:
        if args.baseline == 'replicate':
            assert design_matrix is not None, 'Need design matrix for baseline run "replicate"'
            pairs, singletons, rep_groups = build_sample_pairings(rep_samples, rep_samples, rep_pairs, args.baseline)
            # baseline runs assume a 1-vs-1 setting,
            # so the group size is set to 1
            group1['total'] = 1
            group2['total'] = 1
            group1['samples'] = tuple(sorted(rep_samples))
            group2['samples'] = tuple(sorted(rep_samples))
        elif args.baseline == 'random':
            pairs, singletons, rep_groups = build_sample_pairings(stored_samples, stored_samples,
                                                                  rep_pairs, args.baseline)
            # random baseline run assumes 1-vs-1 setting
            # group size is always 1
            group1['total'] = 1
            group2['total'] = 1
            group1['samples'] = tuple(sorted([p[0] for p in pairs]))
            group2['samples'] = tuple(sorted([p[1] for p in pairs]))
        else:
            raise ValueError('Unexpected value for baseline run: {}'.format(args.baseline))
    elif args.samples or not args.selectgroups:
        # all-vs-all scenario
        if args.samples:
            assert all([s in stored_samples for s in args.samples]), \
                'At least one sample label was not found in the' \
                ' dataset: {}'.format(sorted(stored_samples))
            samples = sorted(args.samples)
        else:
            samples = sorted(stored_samples)
        pairs, singletons, rep_groups = build_sample_pairings(samples, samples, rep_pairs)
        samples1 = sorted(set(p[0] for p in pairs))
        samples2 = sorted(set(p[1] for p in pairs))
        group1['total'] = len(samples1)
        group2['total'] = len(samples2)
        group1['samples'] = tuple(samples1)
        group2['samples'] = tuple(samples2)
    else:
        assert args.selectgroups, 'Control flow error: group selection is not set'
        samples1 = build_sample_group(args.group1, stored_samples, design_matrix)
        samples2 = build_sample_group(args.group2, stored_samples, design_matrix)
        group1['total'] = len(samples1)
        group2['total'] = len(samples2)
        group1['samples'] = tuple(samples1)
        group2['samples'] = tuple(samples2)
        pairs, singletons, rep_groups = build_sample_pairings(samples1, samples2, rep_pairs)

    group1['singletons'] = singletons['group1']
    group2['singletons'] = singletons['group2']
    group1['rep_groups'] = rep_groups['group1']
    group2['rep_groups'] = rep_groups['group2']

    merge_label1, merge_label2 = derive_merge_labels(pairs, design_matrix)
    group1['label'] = merge_label1
    group2['label'] = merge_label2
    metadata = pd.DataFrame(sorted(pairs), columns=['sample1', 'sample2', 'is_replicate'])
    return pairs, group1, group2, metadata


def prepare_baseline_params(args, pairs, group1, group2):
    """
    :param args:
    :param pairs:
    :param group1:
    :param group2:
    :return:
    """
    ka_scan_params, md_params, replicates, chromosomes = None, None, None, None
    binsize = 0
    if args.baseline == 'replicate':
        blen_full = collect_base_lengths(args.dataset, 'full')
        blen_var = collect_base_lengths(args.dataset, 'variable')
        blen_pair = {k: 0 for k in blen_full.keys()}
        with pd.HDFStore(args.dataset, 'r') as hdf:
            md_chrom = hdf[PATH_MD_CHROM]
            binsize = md_chrom.at[md_chrom.index[0], 'binsize']
            chromosomes = md_chrom.index.tolist()
            ka_scan_params = col.defaultdict(dict)
            for s in args.scoring:
                param_scoring = hdf[os.path.join(ROOT_SCORE, s, 'parameters')]
                ka_scoring_params = compute_scan_parameters(md_chrom, param_scoring,
                                                            group1, group2,
                                                            blen_full, 'linear', 'e')
                tmp = pd.DataFrame(list(ka_scoring_params.values()))
                tmp['scoring'] = s
                if md_params is None:
                    md_params = tmp.copy()
                else:
                    md_params = pd.concat([md_params, tmp], axis=0, ignore_index=False)
                ka_scan_params[s]['full'] = ka_scoring_params

                ka_scoring_params = compute_scan_parameters(md_chrom, param_scoring,
                                                            group1, group2,
                                                            blen_var, 'linear', 'e')
                tmp = pd.DataFrame(list(ka_scoring_params.values()))
                tmp['scoring'] = s
                if md_params is None:
                    md_params = tmp.copy()
                else:
                    md_params = pd.concat([md_params, tmp], axis=0, ignore_index=False)
                ka_scan_params[s]['variable'] = ka_scoring_params

                ka_scoring_params = compute_scan_parameters(md_chrom, param_scoring,
                                                            group1, group2,
                                                            blen_pair, 'linear', 'e')
                tmp = pd.DataFrame(list(ka_scoring_params.values()))
                tmp['scoring'] = s
                if md_params is None:
                    md_params = tmp.copy()
                else:
                    md_params = pd.concat([md_params, tmp], axis=0, ignore_index=False)
                ka_scan_params[s]['pairwise'] = ka_scoring_params

        scan_jobs = []
        for s1, s2, is_rep in pairs:
            for c in chromosomes:
                for s in args.scoring:
                    for l in ['full', 'variable', 'pairwise']:
                        scan_jobs.append((args.dataset, s1, s2, c, s, args.rawstats, ka_scan_params[s][l][c], is_rep))
    elif args.baseline == 'random':
        with pd.HDFStore(args.dataset, 'r') as hdf:
            md_chrom = hdf[PATH_MD_CHROM]
            chromosomes = md_chrom.index.tolist()
            blen_full = {row.Index: int(row.bins) for row in md_chrom.itertuples()}
            ka_scan_params = col.defaultdict(dict)
            for s in args.scoring:
                param_scoring = hdf[os.path.join(ROOT_SCORE, s, 'parameters')]
                ka_scoring_params = compute_scan_parameters(md_chrom, param_scoring,
                                                            group1, group2,
                                                            blen_full, 'linear', 'e')
                tmp = pd.DataFrame(list(ka_scoring_params.values()))
                tmp['scoring'] = s
                if md_params is None:
                    md_params = tmp.copy()
                else:
                    md_params = pd.concat([md_params, tmp], axis=0, ignore_index=False)
                ka_scan_params[s]['full'] = ka_scoring_params
        scan_jobs = []
        for s1, s2, is_rep in pairs:
            for c in chromosomes:
                for s in args.scoring:
                    scan_jobs.append((args.dataset, s1, s2, c, s, args.rawstats, ka_scan_params[s]['full'][c], is_rep))
    else:
        raise ValueError('Undefined baseline run: {}'.format(args.baseline))

    return scan_jobs, md_params.reset_index(drop=True), ka_scan_params, binsize


def prepare_scan_params(args, pairs, group1, group2, baselengths, logger):
    """
    :param args:
    :return:
    """
    if args.grouplength == 'adaptive':
        logger.debug('Computing adaptive group lengths for groups: {} vs. {}'.format(group1, group2))
        grouplengths = collect_adaptive_group_size(args, baselengths, group1, group2)
        logger.debug('Done')
    else:
        grouplengths = args.grouplength

    binsize = 0
    ka_scan_params, md_params, chromosomes = dict(), None, None
    with pd.HDFStore(args.dataset, 'r') as hdf:
        for s in args.scoring:
            try:
                _ = hdf[os.path.join(ROOT_SCORE, s, 'matrix')]
            except KeyError:
                sk = [k for k in hdf.keys() if k.startswith(ROOT_SCORE)]
                raise AssertionError('Scoring scheme with name "{}" is not part '
                                     'of dataset {}: {}'.format(s, os.path.basename(args.dataset), sk))
        md_chrom = hdf[PATH_MD_CHROM]
        binsize = md_chrom.at[md_chrom.index[0], 'binsize']

        chromosomes = md_chrom.index.tolist()
        # pre-compute all necessary scan parameters
        for s in args.scoring:
            param_scoring = hdf[os.path.join(ROOT_SCORE, s, 'parameters')]

            ka_scoring_params = compute_scan_parameters(md_chrom, param_scoring,
                                                        group1, group2,
                                                        baselengths, grouplengths, 'e')
            tmp = pd.DataFrame(list(ka_scoring_params.values()))
            tmp['scoring'] = s
            if md_params is None:
                md_params = tmp.copy()
            else:
                md_params = pd.concat([md_params, tmp], axis=0, ignore_index=False)
            ka_scan_params[s] = ka_scoring_params

    scan_jobs = []
    for s1, s2, rep in pairs:
        for c in chromosomes:
            for s in args.scoring:
                scan_jobs.append((args.dataset, s1, s2, c, s, args.rawstats, ka_scan_params[s][c], rep))

    return scan_jobs, md_params.reset_index(drop=True), ka_scan_params, binsize


def _run_cmd_scan(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    logger.debug('Determine sample pairings')
    pairs, group1, group2, md_comp = determine_sample_pairings(args, logger)
    if args.baseline:
        if args.baseline == 'random':
            assert len(args.scoring) == 1, 'Multiple scorings not supported for baseline run' \
                                           ' with strategy "random": {}'.format(args.scoring)
        logger.debug('Preparing parameters for baseline run "{}"'.format(args.baseline))
        scan_jobs, md_params, ka_params, binsize = prepare_baseline_params(args, pairs, group1, group2)
        logger.debug('Create parameter list of size {} for baseline run'.format(len(scan_jobs)))
    else:
        logger.debug('Collecting base lengths for count option "{}"'.format(args.countlength))
        baselengths = collect_base_lengths(args.dataset, args.countlength)
        logger.debug('Base lengths collected')
        scan_jobs, md_params, ka_params, binsize = prepare_scan_params(args, pairs, group1, group2, baselengths, logger)
        logger.debug('Created parameter list of size {} for scan command'.format(len(scan_jobs)))

    scan_results = dict()
    for s in args.scoring:
        scan_results[s] = col.defaultdict(list)

    if args.baseline == 'random':
        num_rand_hsp = 0
        with mp.Pool(args.workers) as pool:
            logger.debug('Start randomized search...')
            while num_rand_hsp < args.numrandom:
                resit = pool.imap_unordered(find_random_hsp, scan_jobs)
                for chrom, segments, scoring in resit:
                    if segments is not None:
                        num_rand_hsp += segments.shape[0]
                        scan_results[scoring][chrom].append(segments)
                    else:
                        print('Empty return')
                logger.debug('Round completed - identified {} random HSPs'.format(num_rand_hsp))
    else:
        i = 0
        total = len(scan_jobs)
        with mp.Pool(args.workers) as pool:
            logger.debug('Start processing...')
            resit = pool.imap_unordered(find_state_hsp, scan_jobs)
            for chrom, segments, scoring in resit:
                i += 1
                logger.debug('Received results for chromosome {} ({}/{})'.format(chrom, i, total))
                if segments is None:
                    logger.debug('No high scoring segments identified')
                    continue
                scan_results[scoring][chrom].append(segments)

    logger.debug('Scanning finished, storing results...')
    os.makedirs(os.path.dirname(os.path.abspath(args.runout)), exist_ok=True)
    merge_segments = []
    with pd.HDFStore(args.runout, 'w') as hdf:
        logger.debug('Saving metadata on comparisons...')
        hdf.put(PATH_MD_COMPARISON, md_comp, format='table')
        logger.debug('Saving metadata on parameters...')
        hdf.put(PATH_MD_COMP_PARAMS, md_params, format='table')
        for s in args.scoring:
            score_results = scan_results[s]
            for chrom, rundata in score_results.items():
                out_path = os.path.join(ROOT_SCAN, s, chrom)
                logger.debug('Writing to path {}'.format(out_path))
                out_data = pd.concat(rundata, axis=0, ignore_index=False)
                out_data.sort_values('nat_score', axis=0,
                                     ascending=False, inplace=True)
                out_data.reset_index(drop=True, inplace=True)
                if args.merge and not args.baseline:
                    # ka_params is structured differently
                    # for baseline comparisons
                    merge_segments.append((out_path, out_data, args.bootstraps,
                                           ka_params[s][chrom],
                                           group1['label'], group2['label'],
                                           args.mergedstats,
                                           group1['total'] * group2['total']))
                hdf.put(out_path, out_data, format='fixed')
            hdf.flush()
        hdf.flush()
        if args.merge and not args.baseline:
            logger.debug('Stored all raw results - merging overlapping segments...')
            with mp.Pool(args.workers) as pool:
                logger.debug('Start processing...')
                resit = pool.imap_unordered(merge_state_hsp, merge_segments)
                for chrom, scoring, mergedata in resit:
                    logger.debug('Received merged segments for chromosome {}'.format(chrom))
                    out_path = os.path.join(ROOT_HSP, scoring, chrom)
                    logger.debug('Writing data to path {}'.format(out_path))
                    hdf.put(out_path, mergedata, format='fixed')
                hdf.flush()

    logger.debug('Output file closed')
    return 0
