# coding=utf-8

import os as os
import re as re

import numpy as np
import pandas as pd

from scidlib.auxiliary import open_mode_heuristic


def get_file_processor(tool):
    """
    :param tool:
    :return:
    """
    funs = {'ChromHMM': process_chromhmm_data,
            'EpiCSeg': process_epicseg_data}
    return funs[tool]


def get_emission_parser(param, tool):
    """
    :param param:
    :param tool:
    :return:
    """
    if param == 'generic':
        return read_state_emissions
    else:
        funs = {'EpiCSeg': read_epicseg_model_emissions,
                'ChromHMM': read_chromhmm_model_emissions}
        return funs[tool]


def process_chromhmm_data(params):
    """
    Processor function - supposed to be used
    with multiprocessing.Pool instance
    :param params:
    :return:
    """
    data_type, smp_label, chrom, chromsize, binsize, fpath = params[:6]
    if data_type == 'state':
        # 'state', label, chrom, size, binsize, fp, state_labels, state_colors
        state_labels = params[6]
        state_colors = params[7]
        states, parsed_info = read_chromhmm_states(fpath, chrom, chromsize,
                                                   binsize, state_labels, state_colors)
        parsed_numbers, parsed_labels, parsed_colors = parsed_info
        data_path = os.path.join(data_type, smp_label, chrom)
        return data_type, data_path, states, (parsed_numbers, parsed_labels, parsed_colors)
    elif data_type == 'posterior':
        df_post, post_chrom = read_chromhmm_posterior(fpath)
        assert post_chrom == chrom, 'Error while reading ChromHMM posterior file {}:' \
                                    ' read chromosome {} from file, expected {}'.format(fpath, post_chrom, chrom)
        new_index = np.arange(0, chromsize // binsize * binsize, binsize, dtype=np.uint32)
        assert len(new_index) == df_post.shape[0], 'Index size mismatch: got {}, need {}'.format(new_index.size,
                                                                                                 df_post.shape[0])
        df_post.index = new_index
        data_path = os.path.join(data_type, smp_label, chrom)
        return data_type, data_path, df_post, None
    else:
        raise ValueError('Unexpected data type: {}'.format(data_type))


def read_chromhmm_posterior(fpath):
    """
    :param fpath:
    :return:
    """
    fopen, fmode = open_mode_heuristic(fpath, True)
    with fopen(fpath, fmode) as table:
        # in ChromHMM post file, first line is sample - chrom
        _, chrom = table.readline().strip().split()
        col_headers = table.readline().strip().split()
        df_post = pd.read_csv(table, sep='\t', header=None,
                              names=col_headers, dtype='float16')
    return df_post, chrom


def _parse_chromhmm_state_line(labels, colors):
    """
    :param labels:
    :param colors:
    :return:
    """
    try:
        rgb_col = int(colors) - 1
    except ValueError:
        rgb_col = 0
    if labels == 'EXTRACT' and rgb_col > 0:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state, label = cols[3].split('_')
            state_num = int(re.search('[0-9]+', state).group(0))
            rgb = cols[rgb_col]
            return s, e, state_num, state, label, rgb
    elif labels == 'EXTRACT' and rgb_col < 1:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state, label = cols[3].split('_')
            state_num = int(re.search('[0-9]+', state).group(0))
            return s, e, state_num, state, label, None
    elif labels != 'EXTRACT' and rgb_col < 1:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state = cols[3]
            state_num = int(re.search('[0-9]+', state).group(0))
            return s, e, state_num, state, None, None
    elif labels != 'EXTRACT' and rgb_col > 0:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state = cols[3]
            state_num = int(re.search('[0-9]+', state).group(0))
            rgb = cols[rgb_col]
            return s, e, state_num, state, None, rgb
    else:
        raise ValueError('Cannot handle label / color combination: {} / {} '.format(labels, colors))
    return parse_line


def read_chromhmm_states(fpath, chrom, chromsize, binsize, labels, colors):
    """
    :param fpath:
    :param chrom:
    :param chromsize:
    :param binsize:
    :param labels:
    :param colors:
    :return:
    """
    num_bins = chromsize // binsize
    states = pd.Series(np.zeros(num_bins, dtype=np.uint8),
                       index=np.arange(0, chromsize // binsize * binsize, binsize, dtype=np.uint32))
    fopen, fmode = open_mode_heuristic(fpath, True)
    chrom_line = re.compile('^' + chrom + '\s')
    found = False
    parse_line = _parse_chromhmm_state_line(labels, colors)
    state_numbers = dict()
    state_labels = dict()
    state_colors = dict()
    with fopen(fpath, fmode) as table:
        for line in table:
            if chrom_line.match(line) is not None:
                found = True
                s, e, state_num, state, label, rgb = parse_line(line)
                # Pandas docs:
                # "When slicing, the start bound is included, AND the stop bound is included."
                states.loc[s:e - binsize] = state_num
                state_numbers[state] = state_num
                state_labels[state] = label
                state_colors[state] = rgb
            else:
                if found:
                    break
                continue
    return states, (state_numbers, state_labels, state_colors)


def read_chromhmm_model_emissions(fpath):
    """
    Read state emission probabilities from
    ChromHMM model file
    :param fpath:
    :return:
    """
    read_buffer = []
    fopen, fmode = open_mode_heuristic(fpath, True)
    with fopen(fpath, fmode) as model:
        for line in model:
            if line and line.startswith('emissionprobs'):
                read_buffer.append(line.strip().split()[1:])
    em_probs = pd.DataFrame(read_buffer, columns=['hidden', 'observed', 'label', 'use', 'prob'])
    em_probs = em_probs.astype({'hidden': np.int8, 'observed': np.int8, 'label': str,
                                'use': np.int8, 'prob': np.float64})
    em_probs = em_probs.loc[em_probs['use'] == 1, :]

    state_nums = np.sort(em_probs['hidden'].unique())
    num_states = state_nums.size

    num_observed = em_probs['observed'].unique().size
    label_observed = np.sort(em_probs['label'].unique())

    norm_em = pd.DataFrame(np.zeros((num_states, num_observed), dtype=np.float64),
                           columns=label_observed, index=state_nums)
    for row in em_probs.itertuples():
        norm_em.loc[row.hidden, row.label] = row.prob
    return norm_em

####################
# EpiCSeg functions
####################


def process_epicseg_data(params):
    """
    Processor function - supposed to be used
    with multiprocessing.Pool instance
    :param params:
    :return:
    """
    data_type, smp_label, chrom, chromsize, binsize, fpath = params[:6]
    if data_type == 'state':
        # 'state', label, chrom, size, binsize, fp, state_labels, state_colors
        state_labels = params[6]
        state_colors = params[7]
        states, parsed_info = read_epicseg_states(fpath, chrom, chromsize,
                                                  binsize, state_labels, state_colors)
        parsed_numbers, parsed_labels, parsed_colors = parsed_info
        data_path = os.path.join(data_type, smp_label, chrom)
        return data_type, data_path, states, (parsed_numbers, parsed_labels, parsed_colors)
    else:
        raise ValueError('Unexpected data type: {}'.format(data_type))


def read_epicseg_states(fpath, chrom, chromsize, binsize, labels, colors):
    """
    :param fpath:
    :param chrom:
    :param chromsize:
    :param binsize:
    :param labels:
    :param colors:
    :return:
    """
    num_bins = chromsize // binsize
    states = pd.Series(np.zeros(num_bins, dtype=np.uint8),
                       index=np.arange(0, chromsize // binsize * binsize, binsize, dtype=np.uint32))
    fopen, fmode = open_mode_heuristic(fpath, True)
    chrom_line = re.compile('^' + chrom + '\s')
    found = False
    parse_line = _parse_epicseg_state_line(labels, colors)
    state_numbers = dict()
    state_labels = dict()
    state_colors = dict()
    with fopen(fpath, fmode) as table:
        for line in table:
            if chrom_line.match(line) is not None:
                found = True
                s, e, state_num, state, label, rgb = parse_line(line)
                # Pandas docs:
                # "When slicing, the start bound is included, AND the stop bound is included."
                states.loc[s:e - binsize] = state_num
                state_numbers[state] = state_num
                state_labels[state] = label
                state_colors[state] = rgb
            else:
                if found:
                    break
                continue
    return states, (state_numbers, state_labels, state_colors)


def _parse_epicseg_state_line(labels, colors):
    """
    :param labels:
    :param colors:
    :return:
    """
    try:
        rgb_col = int(colors) - 1
    except ValueError:
        rgb_col = 0
    if labels == 'EXTRACT' and rgb_col > 0:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state, label = cols[3].split('_')
            state_num = int(re.search('[0-9]+', state).group(0))
            rgb = cols[rgb_col]
            return s, e, state_num, state, label, rgb
    elif labels == 'EXTRACT' and rgb_col < 1:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state, label = cols[3].split('_')
            state_num = int(re.search('[0-9]+', state).group(0))
            return s, e, state_num, state, label, None
    elif labels != 'EXTRACT' and rgb_col < 1:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state = cols[3]
            state_num = int(re.search('[0-9]+', state).group(0))
            return s, e, state_num, state, None, None
    elif labels != 'EXTRACT' and rgb_col > 0:
        def parse_line(line):
            cols = line.strip().split()
            s, e = int(cols[1]), int(cols[2])
            state = cols[3]
            state_num = int(re.search('[0-9]+', state).group(0))
            rgb = cols[rgb_col]
            return s, e, state_num, state, None, rgb
    else:
        raise ValueError('Cannot handle label / color combination: {} / {} '.format(labels, colors))
    return parse_line


def read_epicseg_model_emissions(fpath):
    """
    Read state emission probabilities from
    EpiCSeg model file. Implementation follows
    original EpiCSeg code from github:/epicseg/R/model.R

    #parsing emisP
        emisP <- ifHasField(txt, "emisP", nstates, function(lines){
            emisMat <- parseRows(lines)
            lapply(1:nstates, function(i){
                params <- emisMat[i,]
                r=params[1]
                mus <- params[2:ncol(emisMat)]
                mu <- sum(mus)
                if (mu==0) { ps <- rep(1/length(marks), length(marks))
                } else ps <- mus/mu
                list(mu=mu, r=r, ps=ps)

    :param fpath:
    :return:
    """
    fopen, fmode = open_mode_heuristic(fpath, True)
    with fopen(fpath, fmode) as model:
        line = model.readline().strip()
        assert line.startswith('nstates'), \
            'EpiCSeg model file invalid - first entry is not "nstates": {}'.format(line)
        num_states = int(model.readline())
        _ = model.readline()
        obs_states = model.readline().strip().split()
        num_obs = len(obs_states)
        em_start = model.readline().strip()
        assert em_start.startswith('emisP'), \
            'Expected start of emission probability matrix, found: {}'.format(em_start)
        em_buffer = []
        while 1:
            line = model.readline().strip()
            if line.startswith('transP') or not line.strip():
                break
            em_buffer.append(line.split()[1:])
        assert len(em_buffer) == num_states, \
            'Expected {} lines with emission probabilities, read only {}' \
            ' - file {} invalid'.format(num_states, len(em_buffer), fpath)
        em_mat = pd.DataFrame(em_buffer, columns=obs_states,
                              index=np.arange(1, num_states + 1, 1, dtype=np.int8))
        em_mat = em_mat.astype(np.float64)
        row_sum = em_mat.sum(axis=1)
        if (row_sum == 0).any():
            zero_rows = row_sum.index[row_sum == 0]
            unif_prob = pd.Series([1/num_obs for _ in range(num_obs)],
                                  index=obs_states, dtype=np.float64)
            for row in zero_rows:
                em_mat.loc[row, :] = unif_prob

            em_mat.loc[(row_sum > 0), :] = em_mat.loc[(row_sum > 0), :].divide(row_sum[row_sum > 0], axis=0)
        else:
            em_mat = em_mat.divide(row_sum, axis=0)
    assert np.allclose(em_mat.sum(axis=1).values, 1), 'Normalization of emission probabilities failed'
    return em_mat


def read_state_emissions(fpath):
    """
    This is supposed to be a generic tsv table,
    so not special function depending on segmentation format.
    :param fpath:
    :return:
    """
    df = pd.read_csv(fpath, sep='\t', index_col=False, header=0)
    states = map(lambda x: int(re.search('[0-9]+', x).group(0)), df.iloc[:, 0])
    df = df.iloc[:, 1:].copy()
    df.index = list(states)
    df.index.name = 'state'
    if df.index.size == df.columns.size:
        assert (df.index != df.columns).any(), 'Rows and column names are identical in state emission matrix - ' \
                                               ' this looks like a state transition matrix'
    return df
