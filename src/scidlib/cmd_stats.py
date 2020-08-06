# coding=utf-8

import os as os
import collections as col
import itertools as itt
import multiprocessing as mp

import numpy as np
import pandas as pd

from scidlib import ROOT_STAT_COUNT_COMP, ROOT_STAT, \
    ROOT_STAT_COUNT_TRANS_SNG, ROOT_STAT_COUNT_TRANS_REP, \
    PATH_MD_DESIGN, PATH_MD_CHROM, PATH_MD_STATE, PATH_MD_INPUT, \
    PATH_MD_TRANS_SNG, PATH_MD_TRANS_REP, PATH_MD_BASEFREQ, \
    ROOT_STAT_AGREE_COUNT, ROOT_STAT_AGREE_SCORE
from scidlib.sample_handling import get_replicate_pairs, get_nonreplicate_pairs


def add_stats_cmd_parser(subparsers):
    """
    :param subparsers:
    :return:
    """
    parser_stats = subparsers.add_parser('stats',
                                         help='Add statistics to dataset.')

    grp = parser_stats.add_argument_group('Input/output paths')
    grp.add_argument('--sciddo-data', '-d', type=str, dest='dataset', required=True,
                     help='Path to input SCIDDO dataset in HDF5 format. Computed'
                          ' statistics will be added to the dataset, there is no'
                          ' separate output file.')

    grp = parser_stats.add_argument_group('Statistics')
    grp.add_argument('--counts', '-c', action='store_true', default=False, dest='counts',
                     help='Compute the following count statistics:'
                          ' - state composition per sample'
                          ' - state transitions between replicate pairs'
                          ' - state transitions between non-replicate pairs (singletons)'
                          ' All statistics are stored under the root path {}'
                          ' in the SCIDDO dataset. Default: FALSE'.format(ROOT_STAT))

    grp.add_argument('--agreement', '-a', action='store_true', default=False, dest='agree',
                     help='Compute state agreement score between all samples in the'
                          ' dataset. The state agreement score is defined as the number'
                          ' of genomic bins with identical state assignment divided by'
                          ' the number of total genomic bins (summarized over all'
                          ' chromosomes in the dataset).'
                          ' DEFAULT: FALSE')

    grp.add_argument('--force', '-f', action='store_true', default=False, dest='force',
                     help='Force re-computation of statistics. Otherwise, if a statistic'
                          ' is detected in the SCIDDO dataset, its computation is skipped.'
                          ' Default: false')

    parser_stats.set_defaults(execute=_run_cmd_stats)
    return subparsers


def get_state_counts(params):
    """
    :param params:
    :return:
    """
    fpath, load_key, sample = params
    chrom = os.path.split(load_key)[-1]
    with pd.HDFStore(fpath, 'r') as hdf:
        data = hdf[load_key]
        counts = col.Counter(data.values)
    return sample, chrom, counts


def get_transition_count(params):
    """
    :param params:
    :return:
    """
    s1, s2, chrom, fp = params
    with pd.HDFStore(fp, 'r') as hdf:
        data1 = hdf['/state/{}/{}'.format(s1, chrom)]
        data2 = hdf['/state/{}/{}'.format(s2, chrom)]
    counter = col.Counter()
    counter['total'] = data1.size

    unequal_idx = np.array(data1.values != data2.values, dtype=np.int8)
    var_pos = unequal_idx.sum()
    counter['variable'] = unequal_idx.sum()
    counter['identical'] = data1.size - var_pos

    val_counter = col.Counter((u, v) for u, v in zip(data1.values, data2.values))
    counter.update(val_counter)
    return s1, s2, chrom, counter


def compute_state_compositions(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    logger.debug('Count statistics: state composition per sample')
    root_path = ROOT_STAT_COUNT_COMP
    with pd.HDFStore(args.dataset, 'a') as hdf:
        state_data_keys = [k for k in hdf.keys() if k.startswith('/state')]
        state_samples = [k.split('/')[2] for k in state_data_keys]
        count_data_keys = [k for k in hdf.keys() if k.startswith(root_path)]
        count_samples = [k.split('/')[4] for k in count_data_keys]
        to_do_samples = set(state_samples) - set(count_samples)
        params = []
        for state_key, state_sample in zip(state_data_keys, state_samples):
            if state_sample in to_do_samples or args.force:
                params.append((args.dataset, state_key, state_sample))
        if not params:
            logger.debug('All state composition counts already computed - exiting')
        else:
            logger.debug('Created parameter list of length {} to process'.format(len(params)))
            state_counts_wg = col.defaultdict(col.Counter)
            chrom_keys = col.defaultdict(list)
            state_counts_chrom = dict()
            with mp.Pool(args.workers) as pool:
                resit = pool.imap_unordered(get_state_counts, params)
                for sample, chrom, counts in resit:
                    state_counts_wg[sample].update(counts)
                    state_counts_chrom[sample, chrom] = counts
                    chrom_keys[sample].append((sample, chrom))
            logger.debug('All counts collected, building data frames...')
            for sample, wg_counts in state_counts_wg.items():
                df_rows = []
                row_count = pd.DataFrame.from_records(wg_counts, index=['genome'])
                df_rows.append(row_count)
                for chrom_key in chrom_keys[sample]:
                    _, chrom = chrom_key
                    chrom_count = state_counts_chrom[chrom_key]
                    row_count = pd.DataFrame.from_records(chrom_count, index=[chrom])
                    df_rows.append(row_count)
                df_counts = pd.concat(df_rows, axis=0, ignore_index=False)
                df_counts.fillna(value=0, inplace=True)
                df_counts = df_counts.astype(np.int32, copy=True)
                df_counts.sort_index(axis=0, inplace=True)
                out_path = os.path.join(root_path, sample)
                hdf.put(out_path, df_counts, format='fixed')
                logger.debug('Stored counts {}'.format(out_path))
    logger.debug('Count statistics: state composition per sample - done')
    return None


def build_count_table(counts, state_numbers):
    """
    :param counts:
    :param state_numbers:
    :return:
    """
    x = state_numbers.size
    count_table = pd.DataFrame(np.zeros((x, x), dtype=np.uint32),
                               index=state_numbers, columns=state_numbers)
    for n in state_numbers:
        for m in state_numbers:
            count_table.loc[n, m] = counts[n, m]
    return count_table


def build_trans_count_metadata(chrom_counts):
    """
    This metadata table holds the information
    how many positions were variable among
    all comparisons. The "variable" information
    is later used as length normalization factor
    during the KA parameter estimation.

    :param chrom_counts:
     :type: dict
    :return:
     :rtype: Pandas dataframe
    """
    df_idx = []
    df_rows = []
    for chrom, counts in chrom_counts.items():
        df_idx.append(chrom)
        df_rows.append([counts['total'], counts['identical'], counts['variable']])
    df = pd.DataFrame(df_rows, index=df_idx, columns=['total', 'identical', 'variable'])
    return df


def generate_all_sample_pairs(data_keys):
    """
    :param data_keys:
    :return:
    """
    state_keys = [k for k in data_keys if k.startswith('/state')]
    samples = [k.split('/')[2] for k in state_keys]
    sample_pairs = list(itt.combinations(samples, 2))
    return sorted(sample_pairs)


def compute_pair_state_transitions(args, logger):
    """
    Count state transitions between sample pairs
    :param args:
    :param logger:
    :return:
    """
    logger.debug('Count statistics: state transitions between sample pairs')
    root_sng = ROOT_STAT_COUNT_TRANS_SNG
    root_rep = ROOT_STAT_COUNT_TRANS_REP
    with pd.HDFStore(args.dataset, 'a') as hdf:
        try:
            md_design = hdf[PATH_MD_DESIGN]
            raw_paths = [root_rep, root_sng]
            gen_pairs = [get_replicate_pairs, get_nonreplicate_pairs]
            md_trans_paths = [PATH_MD_TRANS_REP, PATH_MD_TRANS_SNG]
        except KeyError:
            logger.warning('No design matrix in dataset - treating all samples'
                           ' as singletons, resulting in an all-vs-all comparison.')
            md_design = None
            raw_paths = [root_sng]
            sample_pairs = generate_all_sample_pairs(list(hdf.keys()))
            gen_pairs = [lambda x: sample_pairs]
            md_trans_paths = [PATH_MD_TRANS_SNG]

        md_chrom = hdf[PATH_MD_CHROM]
        chrom_names = md_chrom.index.values

        md_states = hdf[PATH_MD_STATE]
        state_numbers = np.sort(md_states['number'].values)

        logger.debug('Identified {} chromosomes in dataset'.format(chrom_names.size))
        for path, get_pairs, md_path in zip(raw_paths, gen_pairs, md_trans_paths):
            if not args.force:
                try:
                    gw_total_path = os.path.join(path, 'genome')
                    _ = hdf[gw_total_path]
                    logger.debug('Detected path {} in dataset, skipping computation'.format(gw_total_path))
                    continue
                except KeyError:
                    pass
            sample_pairs = get_pairs(md_design)
            comp = os.path.split(path)[-1]
            if len(sample_pairs) == 0:
                logger.warning('Identified 0 sample pairs for comparison {} - is that correct? Skipping to next...'.format(comp))
                continue
            logger.debug('Identified {} sample pairs for comparison: {}'.format(len(sample_pairs), comp))
            # append individual chromosomes and path to data file
            # to create final list of parameters
            process_params = []
            for p1, p2 in sample_pairs:
                for c in chrom_names:
                    process_params.append((p1, p2, c, args.dataset))
            total_jobs = len(process_params)
            logger.debug('Created parameter list of size {} to process'.format(total_jobs))
            chrom_counts = col.defaultdict(col.Counter)
            with mp.Pool(args.workers) as pool:
                resit = pool.imap_unordered(get_transition_count, process_params)
                for s1, s2, chrom, counts in resit:
                    chrom_counts[chrom].update(counts)
                    chrom_counts['genome'].update(counts)
            logger.debug('All sample pairs processed')
            total_pos = chrom_counts['genome']['total']
            logger.debug('Total positions processed: {}'.format(total_pos))
            pct_identical = np.round((chrom_counts['genome']['identical'] / total_pos) * 100, 2)
            pct_variable = np.round((chrom_counts['genome']['variable'] / total_pos) * 100, 2)
            logger.debug('Thereof, identical positions: {} ({}%)'.format(chrom_counts['genome']['identical'], pct_identical))
            logger.debug('Thereof, variable positions: {} ({}%)'.format(chrom_counts['genome']['variable'], pct_variable))
            md_counts = build_trans_count_metadata(chrom_counts)

            # create and store raw total
            # and pairwise counts for later
            # normalization - these matrices
            # are not necessarily symmetric
            for chrom, counts in chrom_counts.items():
                count_table = build_count_table(counts, state_numbers)
                if chrom == 'genome':
                    count_total = count_table.sum(axis=1).sum()
                    assert count_total == total_pos, 'Missing counts, expected {} but stored {}'.format(total_pos, count_total)
                out_path = os.path.join(path, chrom)
                hdf.put(out_path, count_table, format='fixed')
                logger.debug('Stored transition counts for comparison / chromosome: {}'.format(out_path))

            hdf.put(md_path, md_counts, format='fixed')
            logger.debug('Metadata stored under {}'.format(md_path))
            hdf.flush()
    logger.debug('Count statistics: state transitions between sample pairs - done')
    return None


def compute_pairwise_agreement(params):
    """
    :param params:
    :return:
    """
    fpath, s1, s2, chrom = params
    with pd.HDFStore(fpath, 'r') as hdf:
        states1 = hdf['/state/{}/{}'.format(s1, chrom)]
        states2 = hdf['/state/{}/{}'.format(s2, chrom)]
        agreement = (states1 == states2).sum()
        total = states1.size
        infos = {'total': total, 'identical': agreement}
    return s1, s2, infos


def compute_state_agreement(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    md_inputs, md_chrom = None, None
    with pd.HDFStore(args.dataset, 'r') as hdf:
        try:
            md_inputs = hdf[PATH_MD_INPUT]
            md_chrom = hdf[PATH_MD_CHROM]
        except KeyError:
            raise KeyError('Could not extract sample and chromosome information from dataset {} - '
                           'is this a valid SCIDDO dataset?'.format(args.dataset))
    sample_pairs = [(s1, s2) for s1, s2 in itt.combinations(md_inputs.index, 2)]
    params = [(args.dataset, t[0][0], t[0][1], t[1]) for t in itt.product(sample_pairs, md_chrom.index)]
    logger.debug('Created parameter list of size {} to process'.format(len(params)))

    expected_total = md_chrom['bins'].sum()
    res_cache = col.defaultdict(col.Counter)
    with mp.Pool(args.workers) as pool:
        resit = pool.imap_unordered(compute_pairwise_agreement, params)
        for s1, s2, counts in resit:
            res_cache[(s1, s2)].update(counts)
    logger.debug('Pairwise state agreement computed, finalizing cached data')
    counts = pd.DataFrame(np.zeros((md_inputs.index.size, md_inputs.index.size), dtype=np.int32),
                          columns=md_inputs.index, index=md_inputs.index)
    for (s1, s2), vals in res_cache.items():
        assert vals['total'] == expected_total, 'Total number of bins not ' \
                                                'as expected: {} vs {} for samples' \
                                                ' {} and {}'.format(vals['total'],
                                                                    expected_total,
                                                                    s1, s2)
        counts.loc[s1, s2] = vals['identical']
        counts.loc[s2, s1] = vals['identical']
        counts.loc[s1, s1] = vals['total']
        counts.loc[s2, s2] = vals['total']

    with pd.HDFStore(args.dataset, 'a') as hdf:
        hdf.put(ROOT_STAT_AGREE_COUNT, counts, format='fixed')
        scores = counts / expected_total
        hdf.put(ROOT_STAT_AGREE_SCORE, scores, format='fixed')
    logger.debug('Agreement statistics stored')
    return None


def compute_background_frequencies(args, logger):
    """
    Given all state composition counts for
    all samples/chromosomes, this now computes
    state background frequencies for the entire
    dataset

    :param args:
    :param logger:
    :return:
    """
    logger.debug('Computing state background frequencies for entire dataset')

    all_freqs = None
    with pd.HDFStore(args.dataset, 'a') as hdf:
        load_keys = [k for k in hdf.keys() if k.startswith(ROOT_STAT_COUNT_COMP)]
        for k in load_keys:
            if all_freqs is None:
                all_freqs = hdf[k]
            else:
                tmp = hdf[k]
                assert tmp.shape == all_freqs.shape, \
                    'Structural mismatch between datasets: {} ({} vs {})'.format(k, tmp.shape, all_freqs.shape)
                all_freqs += tmp
        # compute state base frequencies per chromosome
        # as Karlin-Altschul parameter estimates are also made
        # per chromosome
        bg_freq = all_freqs.div(all_freqs.sum(axis=1), axis=0)
        assert np.allclose(bg_freq.sum(axis=1), 1., atol=1e-6), \
            'Background frequencies do not sum to one within tolerance: {}'.format(bg_freq.sum(axis=1))
        hdf.put(PATH_MD_BASEFREQ, bg_freq, format='fixed')
    logger.debug('Dataset-wide state background frequencies computed')
    return None


def _run_cmd_stats(args, logger):
    """
    :param args:
    :return:
    """
    # since this command is supposed to
    # add data to an existing dataset,
    # check that it really does first...
    assert os.path.isfile(args.dataset), 'Path to SCIDDO dataset invalid: {}'.format(args.dataset)
    if args.counts:
        logger.debug('Computing counting statistics...')
        compute_state_compositions(args, logger)
        compute_pair_state_transitions(args, logger)
        compute_background_frequencies(args, logger)
    if args.agree:
        logger.debug('Computing agreement statistic...')
        compute_state_agreement(args, logger)

    logger.debug('All requested statistics computed - exiting...')
    return 0
