
import os as os
import sys as sys
import re as re
import collections as col
import multiprocessing as mp

import pandas as pd
import numpy as np
import numpy.ma as ma

from scidlib import PATH_MD_STATE, PATH_MD_CHROM, \
    ROOT_SCORE, PATH_MD_COMPARISON, ROOT_HSP, ROOT_SCAN, \
    BED_LIMIT, PATH_MD_COMP_PARAMS
from scidlib.auxiliary import sort_columns_bed_order


def add_dump_cmd_parser(subparsers):
    """
    :param subparsers:
    :return:
    """
    parser_dump = subparsers.add_parser('dump',
                                        help='Dump HDF datasets to text format or to stdout.')

    grp = parser_dump.add_argument_group('Input/output paths')
    grp.add_argument('--data-file', '-d', type=str, dest='datafile', required=True,
                     help='Specify path to data file. The expected type of the data file'
                          ' (e.g., a SCIDDO dataset or a run file) depends on which'
                          ' information should be dumped.')
    grp.add_argument('--support-file', '-s', type=str, dest='supportfile', default='',
                     help='Specify path to support file required for some operations.'
                          ' DEFAULT: <empty>')
    grp.add_argument('--output', '-o', type=str, dest='output', default='stdout',
                     help='Specify path to (i) output file, (ii) path to a folder'
                          ' in case you want to dump all metadata from a data file,'
                          ' or (iii) stdout.')

    grp = parser_dump.add_argument_group('General parameters')
    grp.add_argument('--data-type', '-type', type=str, required=True, dest='datatype',
                     choices=['metadata', 'states', 'segments', 'raw',
                              'scores', 'dynamics', 'transitions'])
    grp.add_argument('--scoring', '-sc', type=str, default='auto', dest='scoring',
                     help='Specify scoring name. Necessary for data types'
                          ' "segments", "scores", "dynamics" and "transitions".'
                          ' The default value "auto" can be used if - and only if -'
                          ' there is exactly one scoring included in the dataset,'
                          ' which can be selected automatically.'
                          ' DEFAULT: auto')
    grp.add_argument('--threshold', '-t', type=float, default=1., dest='threshold',
                     help='Specify upper threshold on Expect value of high scoring segments.'
                          ' This value is necessary for data type "segments" and "dynamics".'
                          ' If statistics were also computed for raw scan results, the threshold'
                          ' will be likewise applied when raw results are dumped via data type "raw".'
                          ' DEFAULT: 1')
    grp.add_argument('--limit-bed-output', '-lim', action='store_true', default=False, dest='limitbed',
                     help='Limit BED output (e.g., HSP segments) to genomic regions and'
                          ' basic statistics about each region (e.g., the Expect value'
                          ' of the HSP).'
                          ' DEFAULT: FALSE')

    grp = parser_dump.add_argument_group('Dump metadata')
    grp.add_argument('--dump-metadata', '-md', type=str, nargs='+', default=['all'], dest='metadata',
                     help='Dump either "all" metadata or just the specified storage path(s)'
                          ' in the HDF file. This proceeds as follows: if "--output" is an'
                          ' existing (!) folder, metadata tables will be dumped'
                          ' to individual files with names derived from their storage'
                          ' paths (with a prefix if specified, see below). If the path'
                          ' specified under "--output" is NOT an existing (!) folder,'
                          ' it will be assumed to be a file to be created and all'
                          ' metadata tables are written into this file (separated by'
                          ' newlines if several tables are dumped).'
                          ' DEFAULT: all')
    grp.add_argument('--metadata-prefix', '-mdp', type=str, default='', dest='mdprefix',
                     help='Specify a prefix for the metadata filenames; the prefix'
                          ' will be prepended as is, i.e., use a separator as needed'
                          ' (such as "prefix_"). Use only characters A-Z, a-z, 0-9,'
                          ' underscore ("_"), dash ("-") and dot (".").'
                          ' DEFAULT: <empty>')

    grp = parser_dump.add_argument_group('Dump chromatin states')
    grp.add_argument('--samples', '-smp', type=str, nargs='+', default=['all'], dest='samples',
                     help='Specify sample label(s) or "all". If chromatin states were annotated'
                          ' with state labels and colors, this information will be automatically'
                          ' added to the dumped segmentation. If the option "--output"'
                          ' is set to an existing (!) folder, data will be dumped to'
                          ' individual files named as <SAMPLE-LABEL>_css.bed.'
                          ' Otherwise, if a single sample is given, the "--output"'
                          ' path is assumed to refer to a file to be created. Otherwise'
                          ' (i.e. several samples are specified, but no existing folder),'
                          ' it is assumed that "--output" is a complete prefix such as'
                          ' /path/to/data/my-css_. This prefix will be completed to'
                          ' /path/to/data/my-css_SAMPLE-LABEL.bed for each sample.'
                          ' DEFAULT: all')

    grp = parser_dump.add_argument_group('Dump score tracks',
                                         description='To dump score tracks (in bedGraph format),'
                                                     ' you need to provide a SCIDDO dataset'
                                                     ' ("--data-file") and a run output file'
                                                     ' ("--support-file"). If "--average" is set'
                                                     ' to False, one score track per comparison'
                                                     ' will be dumped with the following filename:'
                                                     ' <SAMPLE-LABEL1>-vs-<SAMPLE-LABEL2>_<SCORING>-scores.bg'
                                                     ' The path specified via "--output" is then assumed'
                                                     ' to be a folder to be created. If "--average" is'
                                                     ' True, you need to specify a path to a file via "--output"'
                                                     ' that ends in ".bg".')
    grp.add_argument('--average', '-avg', action='store_true', default=False, dest='average',
                     help='If set, compute an average score per genomic bin across all'
                          ' comparisons in the dataset. If you set this option, "--output"'
                          ' has to be a path to a file to be created.'
                          ' DEFAULT: False')
    grp.add_argument('--add-header', '-hd', type=str, default='', dest='addheader',
                     help='Add this header line to the bedGraph output. Note that,'
                          ' to comply with the format specification, you need'
                          ' to set this to "type=bedGraph". Per default, no'
                          ' header is added to increase compatibility with'
                          ' downstream tools.'
                          ' DEFAULT: <empty>')

    grp = parser_dump.add_argument_group('Filter by chromatin dynamics',
                                         description='Filter HSPs by chromatin dynamics, i.e.,'
                                                     ' by state switches observed between samples.')
    grp.add_argument('--from-states', '-from', type=int, default=[], nargs='*', dest='fromstates',
                     help='Specify a space-separated list of state numbers that define'
                          ' the source state. Default: <empty>')
    grp.add_argument('--to-states', '-to', type=int, default=[], nargs='*', dest='tostates',
                     help='Specify a space-separated list of state numbers that define'
                          ' the target state. Default: <empty>')
    grp.add_argument('--add-inverse', '-inv', action='store_true', default=False, dest='addinverse',
                     help='Also filter by the inverse relation, i.e. switch "from" and "to" states'
                          ' between sample groups. Default: False')
    grp.add_argument('--split-segments', '-split', action='store_true', default=False, dest='splitsegments',
                     help='Instead of returning the coordinates of the high scoring segment, return'
                          ' the genomic coordinates of the state transitions. This can be helpful'
                          ' in case of long segments that span multiple smaller regions where the'
                          ' specified state transitions are observed. Default: False')

    grp = parser_dump.add_argument_group('Dump transition count matrix')
    grp.add_argument('--add-state-labels', '-lab', action='store_true', default=False, dest='addstatelabels',
                     help='Add state labels to the output matrix (rows and columns). Please note'
                          ' that this requires the respective annotation to be part of the dataset'
                          ' (usually added during the "convert" command). Default: False')

    parser_dump.set_defaults(execute=_run_cmd_dump)
    return subparsers


def load_filter_data(params):
    """
    :param params:
    :return:
    """
    dataset, rundata, scoring = params[:3]
    s1, s2, chrom = params[3:6]
    with pd.HDFStore(dataset, 'r') as hdf:
        states1 = hdf[os.path.join('state', s1, chrom)]
        states2 = hdf[os.path.join('state', s2, chrom)]
    with pd.HDFStore(rundata, 'r') as hdf:
        segments = hdf[os.path.join(ROOT_HSP, scoring, chrom)]
        segments['chrom'] = chrom
    b1 = segments.at[0, 'start_bp'] // segments.at[0, 'start_bin']
    b2 = segments.at[0, 'end_bp'] // segments.at[0, 'end_bin']
    assert b1 == b2, 'Could not determine size of genomic bins for dataset: {} vs {}'.format(b1, b2)
    binsize = b1
    return states1, states2, s1, s2, segments, binsize


def prepare_indices(states, segments, threshold):
    """
    :param states:
    :param segments:
    :return:
    """
    hsp_cov_index = pd.Series(np.zeros_like(states, dtype=np.int8),
                              index=states.index)
    hsp_lut_index = pd.Series(np.ones_like(states, dtype=np.int32),
                              index=states.index)
    hsp_lut_index *= -1
    segments = segments.loc[(segments['segment_expect'] < threshold), :]
    if segments.empty:
        raise ValueError('No segments left after thresholding')
    for row in segments.itertuples():
        # note that Pandas indexing is right-inclusive
        hsp_cov_index.loc[row.start_bp:row.end_bp] = 1
        hsp_lut_index.loc[row.start_bp:row.end_bp] = row.Index
    if hsp_cov_index.sum() == 0:
        raise ValueError('No HSPs for chromosome {}'.format(segments.at[0, 'chrom']))
    return hsp_cov_index, hsp_lut_index


def extract_overlapping_regions(from_idx, to_idx, hsp_cov, hsp_lut, segments, split):
    """
    :param from_idx: boolean index array marking "from" states
    :param to_idx: boolean index array marking "to" states
    :param hsp_cov: boolean index array indicating HSP (bin) positions
    :param hsp_lut: look-up table for HSPs (stores index values of segments)
    :param segments: dataframe of HSPs
    :param split: split segments into fragments
    :return:
    """
    transition = np.logical_and(from_idx.values, to_idx.values)
    hsp_transition = np.logical_and(transition, hsp_cov.values)
    bp_coords = np.array(hsp_lut.index.tolist(), dtype=np.int64)
    msk_hsp_idx = ma.masked_array(hsp_lut.values, mask=hsp_transition)
    filtered = []
    for idx_window in ma.clump_masked(msk_hsp_idx):
        seg_subset = msk_hsp_idx[idx_window].data
        assert np.allclose(seg_subset, seg_subset[0], atol=0, rtol=0),\
            'Masked regions overlaps several segments'
        seg_idx = int(seg_subset[0])
        seg = segments.loc[seg_idx, :].copy()
        if split:
            start = int(bp_coords[idx_window.start])
            end = int(bp_coords[idx_window.stop - 1])
            seg.loc['start_bp'] = start
            # this should be correct since the
            # coordinates stored were Pandas default
            # right inclusive
            seg.loc['end_bp'] = end
            seg['fragment'] = 1
            filtered.append(seg)
        else:
            seg['fragment'] = 0
            filtered.append(seg)
    df = pd.DataFrame(filtered, index=np.arange(len(filtered)))
    if 'sample1' in df:
        df.drop_duplicates(['chrom', 'start_bp', 'end_bp', 'sample1', 'sample2'], inplace=True)
    elif 'group1' in df:
        df.drop_duplicates(['chrom', 'start_bp', 'end_bp', 'group1', 'group2'], inplace=True)
    else:
        raise ValueError('Unexpected structure of segment dataframe: {}'.format(df.columns))
    return df


def apply_chromatin_dynamics_filter(params):
    """
    :param params:
    :return:
    """
    states1, states2, sample1, sample2, segments, binsize = load_filter_data(params)
    chrom = params[5]
    invert, split, threshold = params[8:]
    hsp_cov, hsp_lut = prepare_indices(states1, segments, threshold)
    from_states, to_states = params[6:8]
    from_idx = states1.isin(from_states)
    to_idx = states2.isin(to_states)

    segments['sample1'] = sample1
    segments['sample2'] = sample2

    out_segments = extract_overlapping_regions(from_idx, to_idx, hsp_cov,
                                               hsp_lut, segments, split)
    if invert:
        from_idx = states2.isin(from_states)
        to_idx = states1.isin(to_states)
        out2_segments = extract_overlapping_regions(from_idx, to_idx, hsp_cov,
                                                    hsp_lut, segments, split)
        if 'sample1' in out2_segments:
            tmp = out2_segments['sample1'].copy()
            out2_segments['sample1'] = out2_segments['sample2']
            out2_segments['sample2'] = tmp
        if 'group1' in out2_segments:
            tmp = out2_segments['group1'].copy()
            out2_segments['group1'] = out2_segments['group2']
            out2_segments['group2'] = tmp

        out_segments = pd.concat([out_segments, out2_segments], axis=0, ignore_index=False)
        out_segments.sort_values(['chrom', 'start_bp', 'end_bp'], ascending=True, inplace=True)
    out_segments.sort_values(['chrom', 'start_bp', 'end_bp'], ascending=True, inplace=True)
    out_segments['#chrom'] = out_segments['chrom']
    out_segments['chromStart'] = out_segments['start_bp']
    out_segments['chromEnd'] = out_segments['end_bp']

    out_segments.sort_values(['chromStart', 'chromEnd'], inplace=True)
    out_segments.reset_index(drop=True, inplace=True)
    out_segments['loc_str'] = 'L' + out_segments.index.astype(str)

    out_segments['rank_pct'] = out_segments['segment_expect'].rank(pct=True, ascending=True).round(decimals=3)
    out_segments['rank_str'] = 'R' + ((out_segments['rank_pct'] * 1000).astype(np.int16).astype(str).str.zfill(4))
    out_segments['name'] = 'HSP_' + chrom + '_' + out_segments['loc_str'] + '_' + out_segments['rank_str']

    out_segments.drop(['chrom', 'start_bp', 'end_bp', 'rank_str', 'loc_str'], axis=1, inplace=True)
    return chrom, out_segments


def filter_by_chromatin_dynamics(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    assert args.scoring, 'You need to provide scoring name via "--scoring" parameter'
    md_comp, md_chrom = None, None
    with pd.HDFStore(args.supportfile, 'r') as hdf:
        try:
            md_comp = hdf[PATH_MD_COMPARISON]
        except KeyError as ke:
            logger.error('No metadata on comparisons -'
                         ' this seems not to be a run output file: {}'.format(args.supportfile))
            raise ke
    with pd.HDFStore(args.datafile, 'r') as hdf:
        try:
            md_chrom = hdf[PATH_MD_CHROM]
        except KeyError as ke:
            logger.error('No metadata on chromosomes -'
                         ' this seems not to be a SCIDDO dataset file: {}'.format(args.datafile))
            raise ke

    filter_params = []
    for row in md_comp.itertuples():
        for chrom in md_chrom.index:
            filter_params.append((args.datafile, args.supportfile, args.scoring,
                                  row.sample1, row.sample2, chrom, args.fromstates, args.tostates,
                                  args.addinverse, args.splitsegments, args.threshold))
    logger.debug('Created argument list of size {} to process'.format(len(filter_params)))
    switches = []
    with mp.Pool(args.workers) as pool:
        resit = pool.imap_unordered(apply_chromatin_dynamics_filter, filter_params)
        for chrom, obj in resit:
            logger.debug('Received {} filtered regions for chromosome {}'.format(obj.shape[0], chrom))
            switches.append(obj)
    switches = pd.concat(switches, axis=0, ignore_index=False)
    col_order = sort_columns_bed_order(switches.columns)
    switches = switches[col_order]
    switches.sort_values(by=['#chrom', 'chromStart', 'chromEnd'], axis=0, inplace=True)
    if args.limitbed:
        bed_columns = BED_LIMIT.split()
        if 'sample1' in switches:
            bed_columns = bed_columns + ['sample1', 'sample2']
        else:
            bed_columns = bed_columns + ['group1', 'group2']
        switches = switches[bed_columns]
    dump_bed_data(switches, args.output, 'w', header=True)

    return 0


def count_state_transitions_per_comparison(params):
    """
    :param params:
    :return:
    """
    states1, states2, sample1, sample2, segments, binsize = load_filter_data(params)
    chrom, threshold = params[5:]

    original_length = states1.size

    try:
        hsp_cov_idx, _ = prepare_indices(states1, segments, threshold)
    except ValueError:
        # can be triggered if, for a given E-value thresholds,
        # no DCDs were detected on the respective chromosome
        return sample1, sample2, chrom + ' (NO DCDs!)', col.Counter()
    hsp_cov_idx = np.array(hsp_cov_idx.values, dtype=np.bool)

    states1 = states1.values[hsp_cov_idx]
    states2 = states2.values[hsp_cov_idx]

    subset_length = states1.size

    assert subset_length < original_length, \
        'Subsetting chromatin state vectors failed: {} vs {}'.format(original_length, subset_length)

    state_pair_counts = col.Counter(zip(states1, states2))

    return sample1, sample2, chrom, state_pair_counts


def count_hsp_state_transitions(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    assert args.scoring, 'You need to provide scoring name via "--scoring" parameter'
    md_comp, md_chrom, scoring_matrix, state_info = None, None, None, None
    with pd.HDFStore(args.supportfile, 'r') as hdf:
        try:
            md_comp = hdf[PATH_MD_COMPARISON]
        except KeyError as ke:
            logger.error('No metadata on comparisons -'
                         ' this seems not to be a run output file: {}'.format(args.supportfile))
            raise ke
    with pd.HDFStore(args.datafile, 'r') as hdf:
        try:
            md_chrom = hdf[PATH_MD_CHROM]
        except KeyError as ke:
            logger.error('No metadata on chromosomes -'
                         ' this seems not to be a SCIDDO dataset file: {}'.format(args.datafile))
            raise ke
        scoring_path = os.path.join(ROOT_SCORE, args.scoring, 'matrix')
        try:
            scoring_matrix = hdf[scoring_path]
        except KeyError as ke:
            logger.error('No scoring matrix found in dataset under path {} -'
                         ' this SCIDDO dataset is not compatible with the supplied'
                         ' SCIDDO run data file (support-file)'.format(scoring_path))
            raise ke
        state_info = hdf[PATH_MD_STATE]

    job_params = []
    for row in md_comp.itertuples():
        for chrom in md_chrom.index:
            job_params.append((args.datafile, args.supportfile, args.scoring,
                               row.sample1, row.sample2, chrom, args.threshold))
    logger.debug('Created argument list of size {} to process'.format(len(job_params)))

    combined_counts = col.Counter()
    with mp.Pool(args.workers) as pool:
        resit = pool.imap_unordered(count_state_transitions_per_comparison, job_params)
        for s1, s2, c, counts in resit:
            logger.debug('Received state transition counts for {} - {} - {}'.format(s1, s2, c))
            combined_counts.update(counts)

    # load scoring matrix as template for output
    scoring_matrix.loc[:] = 0
    trans_count_matrix = scoring_matrix.astype(np.int32)
    for (state1, state2), count in combined_counts.items():
        trans_count_matrix.loc[state1, state2] = count

    if args.addstatelabels:
        trans_count_matrix.index = state_info['description']
        trans_count_matrix.columns = state_info['description'].tolist()

    os.makedirs(os.path.abspath(os.path.dirname(args.output)), exist_ok=True)
    trans_count_matrix.to_csv(args.output, sep='\t', header=True,
                              index=True, index_label='state_transitions')

    return


def dump_segments(args, store_path, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    assert args.scoring, 'You need to provide scoring name via "--scoring" parameter'
    with pd.HDFStore(args.datafile, 'r') as hdf:
        key_filter = os.path.join(store_path, args.scoring)
        load_keys = [k for k in hdf.keys() if k.startswith(key_filter)]
        assert load_keys, \
            'No segments to load from dataset {} for scoring "{}"'.format(args.datafile, args.scoring)
        hsp = []
        for k in load_keys:
            chrom = os.path.split(k)[-1]
            data = hdf[k]
            data['#chrom'] = chrom
            data['chromStart'] = data['start_bp']
            data['chromEnd'] = data['end_bp']
            # note that this naturally holds for raw scan results
            # as the default for the segment expect is -1
            data = data.loc[data['segment_expect'] < args.threshold, :].copy()
            data.sort_values(['chromStart', 'chromEnd'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['loc_str'] = 'L' + data.index.astype(str)
            if np.isclose(data['segment_expect'].values, -1, atol=1e-6).all():
                # Either dumping raw scans or did not request statistics.
                # In any case, ranking by segment expect is pointless,
                # so rank by normalized score
                data['rank_pct'] = data['nat_score_lnorm'].rank(pct=True, ascending=False).round(decimals=3)
            else:
                data['rank_pct'] = data['segment_expect'].rank(pct=True, ascending=True).round(decimals=3)
            data['rank_str'] = 'R' + ((data['rank_pct'] * 1000).astype(np.int16).astype(str).str.zfill(4))
            data['name'] = 'HSP_' + chrom + '_' + data['loc_str'] + '_' + data['rank_str']
            data.drop(['start_bp', 'end_bp', 'rank_str', 'loc_str'], axis=1, inplace=True)
            hsp.append(data)
    hsp = pd.concat(hsp, axis=0, ignore_index=False)
    logger.debug('Collected {} segments with Expect < {}'.format(hsp.shape[0], args.threshold))
    hsp.sort_values(by=['#chrom', 'chromStart', 'chromEnd'], axis=0, inplace=True)
    col_order = sort_columns_bed_order(hsp.columns)
    hsp = hsp[col_order]
    if args.limitbed:
        bed_columns = BED_LIMIT.split()
        if 'sample1' in hsp:
            bed_columns = bed_columns + ['sample1', 'sample2']
        else:
            bed_columns = bed_columns + ['group1', 'group2']
        hsp = hsp[bed_columns]
    dump_bed_data(hsp, args.output, 'w', True)
    return 0


def condense_segmentation(segmentation, binsize, val_type='categorical'):
    """
    :param segmentation:
    :param binsize:
    :param val_type:
    :return:
    """
    col_types = {'categorical': {'chromStart': np.int32,
                                 'chromEnd': np.int32,
                                 'value': np.int8},
                 'continuous': {'chromStart': np.int32,
                                'chromEnd': np.int32,
                                'value': np.float16}}
    starts = [segmentation.index[0]]
    ends = []
    values = []
    last = segmentation[0]
    end = segmentation.index[0]

    for idx, value in segmentation[1:].iteritems():
        if value != last:
            values.append(last)
            end += binsize
            ends.append(end)
            last = value
            starts.append(idx)
        else:
            end += binsize

    ends.append(end + binsize)
    values.append(last)

    assert len(starts) == len(ends) == len(values), \
        'State condensation failed: s {} - e {} - st {}'.format(len(starts),
                                                                len(ends),
                                                                len(values))

    df = pd.DataFrame([starts, ends, values], dtype=None,
                      index=['chromStart', 'chromEnd', 'value'])
    # this is supposed to be dumped
    # as BED-file, need to transpose
    df = df.transpose()

    return df.astype(col_types[val_type])


def dump_sample_states(params):
    """
    :param params:
    :return:
    """
    dataset, load_keys, binsize, sample, out_path, prefix, suffix = params
    load_keys = sorted(load_keys, key=lambda x: x.split('/')[-1])
    if not out_path:
        outfile = 'stdout'
    else:
        outfile = os.path.join(out_path, prefix + sample + suffix)

    filemode = 'w'
    with pd.HDFStore(dataset, 'r') as hdf:
        annot = hdf[PATH_MD_STATE]
        state_names = {row.number: str(row.Index) for row in annot.itertuples()}
        state_colors = {row.number: str(row.color) for row in annot.itertuples()}
        state_labels = {row.number: str(row.description) for row in annot.itertuples()}
        for k in load_keys:
            out_cols = ['#chrom', 'chromStart', 'chromEnd', 'name',
                        'score', 'strand', 'thickStart', 'thickEnd']
            chrom = k.split('/')[-1]
            data = hdf[k]
            table = condense_segmentation(data, binsize)
            table['name'] = table['value'].replace(state_names)
            table['strand'] = '.'
            table['#chrom'] = chrom
            table['thickStart'] = table['chromStart']
            table['thickEnd'] = table['chromEnd']
            table['score'] = (table['chromEnd'] - table['chromStart']).clip(0, 1000)
            if 'no_description' not in state_labels.values():
                table['desc'] = table['value'].replace(state_labels)
                table['name'] = table['name'] + '_' + table['desc']
            if 'no_color' not in state_colors.values():
                table['itemRgb'] = table['value'].replace(state_colors)
                out_cols.append('itemRgb')
            table = table[out_cols].copy()
            table.sort_values(by=['chromStart', 'chromEnd'], inplace=True, axis=0)
            dump_bed_data(table, outfile, filemode, filemode == 'w')
            filemode = 'a'
    return sample, outfile


def dump_bed_data(data, dest, filemode, header, fformat=None):
    """
    :param data:
    :param dest:
    :param filemode:
    :return:
    """
    if dest == 'stdout':
        data.to_csv(sys.stdout, sep='\t', header=header,
                    index=False, float_format=fformat)
    else:
        path, name = os.path.split(dest)
        abspath = os.path.abspath(path)
        os.makedirs(abspath, exist_ok=True)
        data.to_csv(dest, sep='\t', header=header, index=False,
                    mode=filemode, line_terminator='\n', float_format=fformat)
    return


def dump_comparison_scores(params):
    """
    :param params:
    :return:
    """
    dataset, out_path, scoring, add_header, samples = params
    scoring_name = scoring.split('/')[2]
    binsize = 0
    filemode = 'w'
    outputfile = None
    with pd.HDFStore(dataset, 'r') as hdf:
        score_matrix = hdf[scoring]
        md_chrom = hdf[PATH_MD_CHROM]
        chromosomes = sorted(md_chrom.index.tolist())

        for chrom in chromosomes:

            chrom_data = None
            for s1, s2 in samples:
                s1_states = hdf[os.path.join('state', s1, chrom)]
                s2_states = hdf[os.path.join('state', s2, chrom)]
                bs1 = s1_states.index[1] - s1_states.index[0]
                bs2 = s2_states.index[1] - s1_states.index[0]
                assert bs1 == bs2, 'Could not infer bin size from state segmentations: {} vs {}'.format(bs1, bs2)
                binsize = bs1
                scores = pd.Series(score_matrix.lookup(s1_states, s2_states),
                                   index=s1_states.index)
                if chrom_data is None:
                    chrom_data = scores
                else:
                    chrom_data = pd.concat([chrom_data, scores], axis=1, ignore_index=False)
            if isinstance(chrom_data, pd.DataFrame):
                num_comp = chrom_data.shape[1]
                chrom_data = chrom_data.sum(axis=1)
                chrom_data /= num_comp
                chrom_data = chrom_data.round(decimals=2)
            chrom_data = condense_segmentation(chrom_data, binsize, 'continuous')
            chrom_data['chrom'] = chrom
            chrom_data['value'] = chrom_data['value'].round(decimals=2)
            chrom_data = chrom_data[['chrom', 'chromStart', 'chromEnd', 'value']]

            if out_path == 'stdout':
                outputfile = out_path
                if add_header:
                    _ = sys.stdout.write(add_header + '\n')
            else:
                if out_path.endswith('.bg'):
                    outputfile = out_path
                    if add_header:
                        with open(out_path, 'w') as dump:
                            _ = dump.write(add_header + '\n')
                        filemode = 'a'
                        add_header = ''
                else:
                    outname = '{}-vs-{}_{}-scores.bg'.format(s1, s2, scoring_name)
                    outputfile = os.path.join(out_path, outname)
                    if add_header:
                        with open(outputfile, 'w') as dump:
                            _ = dump.write(add_header + '\n')
                        filemode = 'a'
                        add_header = ''
            dump_bed_data(chrom_data, outputfile, filemode, False, '%.2f')
            filemode = 'a'
    return outputfile


def dump_score_tracks(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    score_mat, md_comp = '', None
    with pd.HDFStore(args.datafile, 'r') as hdf:
        try:
            score_mat = os.path.join(ROOT_SCORE, args.scoring, 'matrix')
            _ = hdf[score_mat]
        except KeyError as ke:
            logger.error('Scoring matrix "{}" is not contained in dataset {}'.format(args.scoring, args.datafile))
            raise ke

    with pd.HDFStore(args.supportfile, 'r') as hdf:
        try:
            md_comp = hdf[PATH_MD_COMPARISON]
        except KeyError as ke:
            logger.error('No metadata on comparisons -'
                         ' this seems not to be a run output file: {}'.format(args.supportfile))
            raise ke
    if args.average:
        if args.output != 'stdout':
            # output has to be a file path
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            assert args.output.endswith('.bg'), \
                'Failed to detect ".bg" file extension for output path: {}'.format(args.output)
        params = [(args.datafile, args.output, score_mat, args.addheader,
                   [(row.sample1, row.sample2) for row in md_comp.itertuples()])]
    else:
        assert args.output != 'stdout', \
            'Cannot write individual comparisons to stdout -' \
            ' please specify a different output path (folder)'
        # output has to be a directory path
        os.makedirs(args.output, exist_ok=True)
        params = []
        for row in md_comp.itertuples():
            params.append((args.datafile, args.output, score_mat, args.addheader, [(row.sample1,
                                                                                    row.sample2)]))

    with mp.Pool(min(args.workers, len(params))) as pool:
        resit = pool.imap_unordered(dump_comparison_scores, params)
        for path in resit:
            logger.debug('Score track dumped to file {}'.format(path))
    return 0


def dump_state_segmentations(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    dump_states = col.defaultdict(list)
    with pd.HDFStore(args.datafile, 'r') as hdf:
        for k in hdf.keys():
            if k.startswith('/state'):
                _, _, smp, chrom = k.split('/')
                if smp in args.samples or 'all' in args.samples:
                    dump_states[smp].append(k)
        chroms = hdf[PATH_MD_CHROM]
        binsize = chroms.at[chroms.index[0], 'binsize']
        if len(dump_states) == 0:
            logger.warning('No samples matched in dataset: {}'.format(args.samples))
            return 0
    path, prefix, suffix = '', '', ''
    if args.output == 'stdout':
        assert len(dump_states) == 1, \
            'Cannot dump state segmentations of several samples to stdout'
    elif os.path.isdir(args.output) or args.output.endswith('/'):
        path = args.output
        prefix = ''
        suffix = '_css.bed'
    elif args.output.endswith('.bed'):
        assert len(dump_states) == 1, \
            'Can only dump one sample per fully specified output file: {}'.format(args.output)
        path = os.path.dirname(args.output)
        os.makedirs(path, exist_ok=True)
        prefix = ''
        suffix = ''
    elif len(dump_states) >= 1:
        path = os.path.dirname(args.output)
        os.makedirs(path, exist_ok=True)
        prefix = os.path.basename(args.output)
        assert re.match('^[\w\-\.]+$', prefix), \
            'Specified prefix does not match required character composition: {}'.format(prefix)
        suffix = '.bed'
    else:
        raise RuntimeError('Unexpected option: {}'.format(args.output))

    params = []
    for k, v in dump_states.items():
        p = args.datafile, tuple(v), binsize, k, path, prefix, suffix
        params.append(p)

    with mp.Pool(min(args.workers, len(dump_states))) as pool:
        resit = pool.imap_unordered(dump_sample_states, params)
        for smp, fpath in resit:
            logger.debug('State segmentation of {} written to {}'.format(smp, fpath))
    return 0


def dump_metadata(args, logger):
    """
    :param args:
    :param logger:
    :return:
    """
    if args.output == 'stdout':
        base_out = sys.stdout
        is_prefix = False
    elif os.path.isdir(args.output):
        if args.mdprefix:
            assert re.match('^[\w\-\.]+$', args.mdprefix), \
                'Specified prefix does not match required character composition: {}'.format(args.mdprefix)
            base_out = os.path.join(args.output, args.mdprefix)
            is_prefix = True
        else:
            base_out = os.path.join(args.output, '')
            is_prefix = False
    else:
        # assume it is a file to be created
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        base_out = args.output
        is_prefix = False
        with open(base_out, 'w') as dump:
            # just create file and open
            # later in "a" mode
            pass
    logger.debug('Start dumping metadata...')
    with pd.HDFStore(args.datafile, 'r') as hdf:
        if 'all' in args.metadata:
            load_keys = [k for k in hdf.keys() if k.startswith('/metadata') or k.startswith(ROOT_SCORE)]
        else:
            norm = [k if k.startswith('/') else '/' + k for k in args.metadata]
            load_keys = [k for k in norm if k in hdf.keys()]
        if not load_keys:
            raise ValueError('Normalizing metadata key(s) failed or key(s) not contained in dataset: {}'.format(args.metadata))
        for k in load_keys:
            table = hdf[k]
            derived_name = k.strip('/').replace('metadata', 'md').replace('/', '_') + '.tsv'
            if is_prefix:
                outputfile = base_out + derived_name
            elif isinstance(base_out, str) and base_out.endswith('/'):
                # is a folder
                outputfile = os.path.join(base_out, derived_name)
            else:
                outputfile = base_out
            if hasattr(outputfile, 'write'):
                if len(load_keys) > 1:
                    _ = outputfile.write('{}\n'.format(k))
                table.to_csv(outputfile, sep='\t', header=True,
                             index=True, index_label='row_index')
                _ = outputfile.write('\n')
            else:
                logger.debug('Dumping to file {}'.format(outputfile))
                if len(load_keys) > 1 and not is_prefix:
                    with open(outputfile, 'a') as dump:
                        _ = dump.write('{}\n'.format(k))
                table.to_csv(outputfile, sep='\t', header=True, index=True,
                             mode='a', index_label='row_index', line_terminator='\n')
                with open(outputfile, 'a') as dump:
                    _ = dump.write('\n')
    logger.debug('All metadata tables dumped')
    return 0


def extract_auto_scoring(fpath, logger, filter_path=None, load_path=None):
    """
    :param fpath:
    :param logger:
    :param filter_path:
    :param load_path:
    :return:
    """
    logger.debug('Scoring set to "auto", extracting from input data: {}'.format(fpath))
    with pd.HDFStore(fpath, 'r') as hdf:
        if load_path is not None:
            metadata = hdf[load_path]
            assert 'scoring' in metadata, 'No column "scoring" in metadata loaded from path {}'.format(load_path)
            all_scorings = metadata['scoring'].unique()
            assert all_scorings.size == 1, 'More than one scoring schema included in run dataset.' \
                                           ' Cannot select automatically: {}'.format(all_scorings.tolist())
            scoring = all_scorings[0]
            logger.debug('Extracted scoring {} from path {}'.format(scoring, load_path))
        elif filter_path is not None:
            all_paths = [k.replace(filter_path, '') for k in hdf.keys() if k.startswith(filter_path)]
            path_checker = col.defaultdict(set)
            for p in all_paths:
                components = p.split('/')
                for level, sub in enumerate(components):
                    if not sub:  # could be empty string
                        continue
                    path_checker[level].add(sub)
            scoring = None
            for level in sorted(path_checker.keys()):
                values = path_checker[level]
                if len(values) == 1:
                    scoring = values.pop()
                    break
            if scoring is None:
                raise ValueError('Could not identify scoring underneath root path {}'.format(filter_path))
            logger.debug('Extracted scoring {} from path {}'.format(scoring, filter_path))
        else:
            scoring = None
            raise ValueError('Both filter and load path are None')
    return scoring


def _run_cmd_dump(args, logger):
    """
    :param args:
    :return:
    """
    if args.datatype == 'metadata':
        dump_scoring = any([x.startswith(ROOT_SCORE.strip('/')) for x in args.metadata])
        dump_scoring |= any([x.startswith(ROOT_SCORE) for x in args.metadata])
        if dump_scoring:
            if args.scoring == 'auto':
                scoring = extract_auto_scoring(
                    args.datafile,
                    logger,
                    filter_path=ROOT_SCORE,
                    load_path=None
                    )
            else:
                scoring = args.scoring
            adapted_paths = []
            for md_path in args.metadata:
                if md_path.startswith(ROOT_SCORE.strip('/')) or md_path.startswith(ROOT_SCORE):
                    if scoring in md_path:
                        if not (md_path.endswith('matrix') or md_path.endswith('parameters')):
                            adapted_paths.append(os.path.join(ROOT_SCORE, scoring, 'matrix'))
                            adapted_paths.append(os.path.join(ROOT_SCORE, scoring, 'parameters'))
                        else:
                            adapted_paths.append(md_path)
                    else:
                        adapted_paths.append(os.path.join(ROOT_SCORE, scoring, 'matrix'))
                        adapted_paths.append(os.path.join(ROOT_SCORE, scoring, 'parameters'))
                else:
                    adapted_paths.append(md_path)
            setattr(args, 'metadata', adapted_paths)

        logger.debug('Dumping metadata')
        dump_metadata(args, logger)
    elif args.datatype == 'states':
        logger.debug('Dumping chromatin states')
        dump_state_segmentations(args, logger)
    elif args.datatype == 'scores':
        if args.scoring == 'auto':
            # supportfile is a run file
            scoring = extract_auto_scoring(args.supportfile, logger,
                                           filter_path=None,
                                           load_path=PATH_MD_COMP_PARAMS)
            setattr(args, 'scoring', scoring)
        logger.debug('Dumping score tracks')
        dump_score_tracks(args, logger)
    elif args.datatype in ['segments', 'raw']:
        if args.scoring == 'auto':
            # datafile is a run file
            scoring = extract_auto_scoring(args.datafile, logger,
                                           filter_path=None,
                                           load_path=PATH_MD_COMP_PARAMS)
            setattr(args, 'scoring', scoring)
        if args.datatype == 'segments':
            logger.debug('Dumping HSP segments')
            path_prefix = ROOT_HSP
        elif args.datatype == 'raw':
            logger.debug('Dumping raw scan results')
            path_prefix = ROOT_SCAN
        else:
            raise ValueError('Unexpected data type: {}'.format(args.datatype))
        dump_segments(args, path_prefix, logger)
    elif args.datatype == 'dynamics':
        if args.scoring == 'auto':
            # supportfile is a run file
            scoring = extract_auto_scoring(args.supportfile, logger,
                                           filter_path=None,
                                           load_path=PATH_MD_COMP_PARAMS)
            setattr(args, 'scoring', scoring)
        logger.debug('Dumping HSPs filtered by chromatin dynamics')
        filter_by_chromatin_dynamics(args, logger)
    elif args.datatype == 'transitions':
        if args.scoring == 'auto':
            # supportfile is a run file
            scoring = extract_auto_scoring(args.supportfile, logger,
                                           filter_path=None,
                                           load_path=PATH_MD_COMP_PARAMS)
            setattr(args, 'scoring', scoring)
        logger.debug('Dumping state transitions in HSP regions')
        count_hsp_state_transitions(args, logger)
    else:
        pass
    logger.debug('All data dumped - exiting...')
    return 0
