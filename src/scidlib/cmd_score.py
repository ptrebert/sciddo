# coding=utf-8

import os as os
import re as re
import itertools as itt
import copy as cp
import argparse as argp

import numpy as np
import scipy.special as scisp
import pandas as pd

import karlin as karlin
from scidlib import PATH_MD_STATE, PATH_MD_EMISSION, \
    PATH_MD_CHROM, PATH_MD_TRANS_SNG, ROOT_SCORE, PATH_MD_BASEFREQ, \
    PATH_MD_DESIGN, ROOT_STAT_COUNT_TRANS_REP
from scidlib.statistics import jensen_shannon_divergence as jsd
from scidlib.statistics import compute_score_probabilities


def add_score_cmd_parser(subparsers):
    """
    :param subparsers:
    :return:
    """
    parser_score = subparsers.add_parser('score',
                                         help='Add scoring schemes to dataset. Note that running'
                                              ' the statistics sub-command with the option "--counting"'
                                              ' is required to derive certain scoring schemes based on'
                                              ' the dataset (see below).',
                                         formatter_class=argp.RawTextHelpFormatter)

    grp = parser_score.add_argument_group('Input/output paths')
    grp.add_argument('--sciddo-data', '-d', type=str, dest='dataset', required=True,
                     help='Path to input SCIDDO dataset in HDF5 format. Computed\n'
                          'scoring matrices will be added to the dataset,\n'
                          'there is no separate output file.\n')

    grp = parser_score.add_argument_group('Specify scorings to add')
    grp.add_argument('--add-scoring', '-ads', type=str, nargs='*', default=[], dest='addscoring',
                     help='Add a named scoring scheme to the dataset. Names should only\n'
                          'consist of the characters A-Z, a-z, underscore and numbers 0-9\n'
                          '(but not as first character of the name). Scorings can be added\n'
                          'by specifying a file containing a scoring matrix (see below) or\n'
                          'by selecting one or more of the following default scorings:\n'
                          '\n'
                          '- "emission": Scoring derived from similarity measured as\n'
                          'divergence between state emission probabilities. Requires\n'
                          'a state emission matrix to be present in the dataset.\n'
                          '\n'
                          '- "replicate": Scoring derived from similarity measured as\n'
                          'state transition frequency between biological replicates.\n'
                          'Requires a design matrix to be present in the dataset and several\n'
                          'high quality replicates to give meaningful results.\n'
                          '\n'
                          'For all data-derived scoring schemes, it is necessary to know\n'
                          'the so-called background state, i.e., the state that\n'
                          'represents no detectable signal (either for biological or for\n'
                          'technical reasons). Note that SCIDDO will estimate the\n'
                          'background state from the data during the initial data\n'
                          'conversion. You can explicitly set the background state\n'
                          'if necessary (see below).\n'
                          '\n'
                          'Apart from the default scorings, you can specify an arbitrary\n'
                          'number of names for scoring schemes read from files (see below).\n'
                          'The order of the names has to correspond to the order of the files.\n\n')

    grp.add_argument('--background-state', '-bgs', type=int, default=-1, dest='bgstate',
                     help='Specify the number (not the name!) of the state that\n'
                          'represents regions of no detectable/detected signal (background).\n'
                          'The background state will be estimated based on the data\n'
                          'during the data conversion step. You can overwrite this\n'
                          'by setting this parameter to some other value. The default\n'
                          'of "-1" will trigger to use the estimated state. To see which\n'
                          'state was estimated to be the background state, simply look\n'
                          'at the state metadata table (path: {}) via the dump command.\n\n'
                          'DEFAULT: -1\n\n'.format(PATH_MD_STATE))

    grp.add_argument('--null-obs', '-nob', type=int, default=5, dest='nullobs',
                     help='Specify the percentage of NULL observations that are still\n'
                          'acceptable during replicate-based scoring. A NULL observation\n'
                          'refers to no observed transitions between states AB (or BA).\n'
                          'If there are too many NULL observations, the scoring scheme\n'
                          'derived from the replicates in the dataset is not reliable as\n'
                          'the estimated ratio between observed and expected transitions\n'
                          'is a bad estimate for the true population.\n\n'
                          'DEFAULT: 5 pct.\n\n')

    grp.add_argument('--scoring-file', '-scf', nargs='*', type=str, dest='scoringfile', default=[],
                     help='Specify full path(s) to tab-separated file(s) containing\n'
                          'scoring matrices in one of the formats described below.\n'
                          '=========================\n'
                          'GENERAL INFO:\n'
                          '- positive scores represent state dissimilarity\n'
                          '- negative scores represent state similarity\n'
                          '- scores will be rounded to integers\n'
                          '- if there is no name specified for the scoring scheme via the\n'
                          '  "--add-scoring" parameter above, the scoring name will be identical\n'
                          '  to the file name after stripping the file extension.\n'
                          '- the name has to conform to Python variable naming standards;\n'
                          '  restrict names to A-Z, a-z and 0-9 (not as first character)\n'
                          '  and the underscore "_"\n'
                          '=========================\n'
                          'TABLE / MATRIX format:\n'
                          '- column and row names have to be present (state numbers, not state names)\n'
                          '- the name of the first column can be arbitrary (e.g., "state")\n'
                          '- the matrix has to be complete, i.e., specifying only a lower/upper\n'
                          '  diagonal matrix does not suffice\n'
                          '=========================\n'
                          '3-COLUMN TEXT format:\n'
                          '- no row or column names\n'
                          '- each row lists "state1" "state2" "score"\n'
                          '- only the first 3 columns of the file will be considered\n'
                          '- state pairs with a score of zero can be omitted\n'
                          '- pair AB is identical to BA, i.e., it is sufficient to\n'
                          '  list only one of the two pairs\n\n'
                          'DEFAULT: <empty>\n\n')
    grp.add_argument('--treat-background', '-tbg', type=str, dest='bgtreatment', nargs='*',
                     choices=['penalized', 'ordinary'], default=['penalized'],
                     help='For emission- and replicate-based scorings, the background state can\n'
                          ' be treated as any other chromatin state ("ordinary"), or its\n'
                          ' occurrence can be penalized (receives minimum score for all\n'
                          ' possible state pairings) to reflect the fact that the lack of\n'
                          ' a signal may simply be due to technical artifacts.\n\n'
                          ' Default: penalized')
    grp.add_argument('--replace', '-rep', action='store_true', default=False, dest='replace',
                     help='If the user specifies a scoring and its name already exists,\n'
                          'simply overwrite the existing scoring; otherwise,\n'
                          'this raises a ValueError.\n\n'
                          'DEFAULT: False (raise error)\n')

    parser_score.set_defaults(execute=_run_cmd_score)
    return subparsers


def compute_var_lengths_factors(chroms, var_pos):
    """
    Compute the length normalization factors
    per chromosome. Since chromatin state segmentations
    often show a constant state in the beginning/end
    of the chromosome (all background), the factor
    is calculated only based on the number of genomic bins
    where at least one state transition has been observed
    ("variable bins") in the dataset.
    This number of variable bins per chromosome can be used
    instead of the full length (total number of bins) of
    the chromosome.

    :param chroms: Pandas DataFrame containing chromosome information
    :param var_pos: Pandas DataFrame containing counts of variable bins
                    across the entire dataset
    :return:
    """
    var_lens = dict()
    genome_bins = 0
    genome_total = 0
    genome_var = 0
    for row in chroms.itertuples():
        num_bins = row.bins
        genome_bins += num_bins
        # total is the total number of transitions
        # for the entire dataset - that should always
        # be number of bins times number of comparisons
        # => fac
        fac = var_pos.at[row.Index, 'total'] / num_bins
        genome_total += var_pos.at[row.Index, 'total']
        # this is now the average number of variable
        # positions observed in this dataset and for
        # this chromosome
        this_norm = var_pos.at[row.Index, 'variable'] // fac
        genome_var += var_pos.at[row.Index, 'variable']
        var_lens[row.Index] = np.round(this_norm, 0)
    genome_fac = genome_total / genome_bins
    var_lens['genome'] = np.round(genome_var // genome_fac, 0)
    return var_lens


def read_scoring_scheme(args, logger, score_name, score_file):
    """
    Read scoring matrix from file, either as complete matrix
    or as 3-column text file

    :param args:
    :param logger:
    :param score_name:
    :param score_file:
    :return:
    """
    st_md_path = PATH_MD_STATE
    len_norm_factors = None
    with pd.HDFStore(args.dataset, 'r') as hdf:
        state_table = hdf[st_md_path]
        state_numbers = state_table['number'].values
        num_states = state_numbers.size
        try:
            trans_sng_table = hdf[PATH_MD_TRANS_SNG]
        except KeyError:
            raise AssertionError('State transition counts missing in dataset (path {}).'
                                 ' Please run the "stats" sub-command before proceeding'.format(PATH_MD_TRANS_SNG))
        len_norm_factors = compute_var_lengths_factors(hdf[PATH_MD_CHROM], trans_sng_table)

    table = None  # get rid of IDE warning...
    # check what format the scoring has
    with open(score_file, 'r') as txt:
        header = txt.readline().strip().split('\t')
        txt.seek(0)
        if len(header) == num_states + 1:  # one column per state plus first column (row names)
            table = pd.read_csv(txt, sep='\t', header=0, index_col=0, dtype=np.float32)
            # would like to set the dtype of index for read_csv
            table.index = table.index.astype(np.int16)
            table.columns = table.columns.astype(np.int16)
            table = table.round(0).astype(np.int16)
        elif len(header) >= 3:  # at least 3 column text file
            table = pd.DataFrame(np.zeros((num_states, num_states), dtype=np.float32),
                                 index=sorted(state_numbers), columns=sorted(state_numbers))
            for line in txt:
                if line.strip():
                    s1, s2, sc = line.split('\t')[:3]
                    table.loc[int(s1), int(s2)] = float(sc)
                    table.loc[int(s2), int(s1)] = float(sc)
            table = table.round(0).astype(np.int16)
            table.index.name = 'state'
        else:
            raise ValueError('Cannot determine the format of score file: {}'.format(score_file))
    table.index.name = 'state'
    row_states = set(table.index)
    col_states = set(table.columns)
    assert (table == table.transpose()).all(axis=1).all(), \
        'Read score matrix {} is not symmetric'.format(score_name)
    all_match = row_states - col_states
    assert not all_match,\
        'Row and column state numbers disagree for scoring {}: {}'.format(score_name, all_match)
    all_match = row_states.intersection(col_states.intersection(set(state_numbers)))
    assert len(all_match) == num_states, \
        'Not all state numbers for scoring {}' \
        ' are in the dataset: {} vs {}'.format(score_name, row_states.intersection(col_states), state_numbers)

    store_path = os.path.join(ROOT_SCORE, score_name)
    logger.debug('Storing scoring matrix under root: {}'.format(store_path))

    store_scoring_matrix(args.dataset, store_path, table, len_norm_factors, args.replace)

    return 0


def determine_null_obs_threshold(num_states, percentage):
    """
    Determine the threshold for null observations for
    replicate scoring - if there are too many null
    observations, then the replicate-derived scoring
    is no longer meaningful

    :param num_states:
    :param percentage:
    :return:
    """
    threshold = np.floor(scisp.comb(num_states, 2, exact=True) * (percentage / 100.))
    return threshold


def collect_replicate_transitions(dataset):
    """
    :param dataset: handle on dataset in HDF5 format
    :return: state transition frequencies between replicates
     :rtype: Pandas DataFrame
    """
    # NB: this method could be refactored and turned
    # into a generic "load_transition_freq"...
    load_keys = [k for k in dataset.keys() if k.startswith(ROOT_STAT_COUNT_TRANS_REP)]
    trans = None
    for k in load_keys:
        if trans is None:
            trans = dataset[k]
        else:
            trans += trans
    # make matrix symmetric...
    trans += trans.transpose()
    trans //= 2
    assert (trans == trans.transpose()).all(axis=1).all(), \
        'Replicate transition count matrix not symmetric: {}'.format(trans)
    # and norm to frequencies
    trans_freq = trans / trans.sum(axis=1).sum()
    return trans_freq


def compute_replicate_scoring(args, logger):
    """
    The replicate based scoring follows the derivation of the BLOSUM
    scoring matrices (Henikoff & Henikoff, PNAS 1992). In summary,
    that means that the following log odds score is created based on
    transition frequencies between "aligned" replicates:

    s_ij = 2 * log2( observed freq. / expected (base) freq. )

    :param args:
    :param logger:
    :return:
    """
    logger.debug('Computing replicate scoring...')
    assert args.bgstate > -1, 'Background state must be set to non-default for replicate scoring'
    dm_table_path = PATH_MD_DESIGN
    st_md_path = PATH_MD_STATE
    var_lens = None
    with pd.HDFStore(args.dataset, 'r') as hdf:
        try:
            # No design matrix means no replicates
            # means no transitions frequencies
            # could have been derived
            _ = hdf[dm_table_path]
        except KeyError:
            raise AssertionError('No design matrix detected in dataset: path {} is missing'.format(dm_table_path))
        # get individual replicates to get state counts
        obs_freq = collect_replicate_transitions(hdf)
        state_table = hdf[st_md_path]
        num_states = state_table.shape[0]
        # note here:
        # length normalization factors are calculated
        # based on state transitions between singletons
        # as scans for HSPs will compare singletons
        try:
            trans_sng_table = hdf[PATH_MD_TRANS_SNG]
        except KeyError:
            raise AssertionError('State transition counts missing in dataset (path {}).'
                                 ' Please run the "stats" sub-command before proceeding'.format(PATH_MD_TRANS_SNG))
        var_lens = compute_var_lengths_factors(hdf[PATH_MD_CHROM], trans_sng_table)

    assert args.bgstate in state_table['number'].values, \
        'Specified background state number not in dataset: {}'.format(args.bgstate)

    # init empty matrix
    rep_score = pd.DataFrame(np.zeros((num_states, num_states), dtype=np.float32),
                             index=state_table['number'].sort_values(inplace=False),
                             columns=state_table['number'].sort_values(inplace=False))
    np.seterr(all='raise')
    # Compute expected frequency as for each
    # state as done in Henikoff & Henikoff
    # p_i = q_ii + sum_j!=i (qij / 2)
    states = set(state_table['number'].tolist())
    exp_freq = pd.Series(np.zeros(len(states)), index=sorted(states))
    for s in state_table['number']:
        # select all transitions involving state s (i)
        # and subtract self-transition (ii)
        trans_select = sorted(states - {s})
        # since trans_select is a list, we select
        # all transitions at once
        state_freq = obs_freq.at[s, s] + (obs_freq.loc[s, trans_select] / 2).sum()
        exp_freq[s] = state_freq
    max_null_obs = determine_null_obs_threshold(num_states, args.nullobs)
    null_obs = 0
    for s1, s2 in itt.combinations_with_replacement(state_table['number'].tolist(), r=2):
        try:
            # leading 2: "scaling factor" that is not
            # motivated in the Henikoff & Henikoff paper
            if s1 == s2:
                # s_ii = log2(q_ii / p_i * p_i)
                score = 2 * np.log2(obs_freq.at[s1, s2] / (exp_freq.at[s1] * exp_freq.at[s2]))
            else:
                # s_ii = log2(q_ii / (2 * p_ij * p_ij))
                score = 2 * np.log2(obs_freq.at[s1, s2] / (2 * exp_freq.at[s1] * exp_freq.at[s2]))
        except (FloatingPointError, ZeroDivisionError):
            # if the observed frequency is zero
            # record and break if there are too
            # many such events
            null_obs += 1
            if null_obs > max_null_obs:
                raise RuntimeError('Reached maximal number of NULL observations: {}'.format(max_null_obs))
            logger.warning('ZeroDivision: no observed transitions between {} and {}'.format(s1, s2))
            # It may indeed happen that the observed
            # state transition frequency is zero
            # if the number of replicates is too low or
            # if the two states are really different
            # (a transition would never be observed
            # between biological replicates)
            # Frequency zero will be interpreted
            # as "much lower than expected"
            score = np.nan
        rep_score.loc[s1, s2] = np.round(score, 0)
        rep_score.loc[s2, s1] = np.round(score, 0)
    # state right now:
    # positive entries: observed higher than expected,
    # states are (biologically) similar
    # negative entries: observed lower than expected,
    # states are (biologically) dissimilar
    max_dissim = rep_score.min(axis=1).min()
    # null observations were marked as NaN
    # replace this now with maximal dissimilarity,
    # i.e., the minimal score
    rep_score.replace(np.nan, max_dissim, inplace=True)
    rep_score = rep_score.astype(np.int16)
    # What follows:
    # For some "not so well-defined states" (e.g., TssD)
    # it is possible that these are not "well aligned"
    # in replicates and, thus, this state would not appear
    # to be paired with itself (at least, not often enough
    # to qualify as similar. Hence, it is necessary to adapt
    # the main diagonal as well (i.e., make all states
    # dissimilar to themselves) ; go for max
    # positive score (max similarity) + 1
    max_sim = rep_score.max(axis=1).max() + 1
    rep_score.values[np.diag_indices(rep_score.shape[0])] = max_sim
    # If option is set:
    # transitions from/to background state
    # are penalized for the lack of
    # interpretability
    if args.bgtreatment == 'penalized':
        rep_score.loc[sorted(states), args.bgstate] = max_sim
        rep_score.loc[args.bgstate, sorted(states)] = max_sim
    # Turn scores into the dissimilarities
    # we are interested in
    rep_score *= -1
    assert (rep_score == rep_score.transpose()).all(axis=1).all(), \
        'Replicate scoring matrix not symmetric: {}'.format(rep_score)

    if args.bgtreatment == 'penalized':
        store_path = os.path.join(ROOT_SCORE, 'penrep')
    else:
        store_path = os.path.join(ROOT_SCORE, 'ordrep')
    logger.debug('Storing replicate-based scoring matrix '
                 'with {} background state under root: {}'.format(args.bgtreatment, store_path))

    store_scoring_matrix(args.dataset, store_path, rep_score, var_lens, args.replace)

    return 0


def compute_emission_scoring(args, logger):
    """
    Compute a scoring matrix based on the state
    segmentation model emissions.
    The notion of (dis-) similarity is defined as
    the Jensen-Shannon-Divergence (JSD) between the
    emission probability vectors for all pairs
    of states. Since the JSD is non-negative,
    the resulting matrix of divergences is shifted
    into the negative domain by the mean JSD per row.
    Since the minimal JSD (0.) results from identical
    P and Q (= the most extreme case of similarity),
    this shift moves all similar state pairs (low
    JSD) into the negative domain; all dissimilar
    state pairs (high JSD) remain in the positive domain.

    :param args:
    :param logger:
    :return:
    """
    logger.debug('Computing emission scoring...')
    assert args.bgstate > -1, 'Background state must be set to non-default for emission scoring'

    # these inits just to get rid of IDE warnings...
    var_lens = None
    state_table, em_table, em_dim = None, None, 0
    with pd.HDFStore(args.dataset, 'r') as hdf:
        try:
            em_table = hdf[PATH_MD_EMISSION]
            em_dim = em_table.shape[0]
        except KeyError:
            raise AssertionError('No emission table detected in dataset: path {} is missing'.format(PATH_MD_EMISSION))
        # if the state table was missing, the entire
        # dataset would be corrupt
        state_table = hdf[PATH_MD_STATE]
        try:
            trans_sng_table = hdf[PATH_MD_TRANS_SNG]
        except KeyError:
            raise AssertionError('State transition counts missing in dataset (path {}).'
                                 ' Please run the "stats" sub-command before proceeding'.format(PATH_MD_TRANS_SNG))
        var_lens = compute_var_lengths_factors(hdf[PATH_MD_CHROM], trans_sng_table)
    assert args.bgstate in state_table['number'].values, \
        'Specified background state number not in dataset: {}'.format(args.bgstate)

    # prepare empty dataframe for scoring
    em_score = pd.DataFrame(np.zeros((em_dim, em_dim), dtype=np.float32),
                            index=state_table['number'].sort_values(inplace=False),
                            columns=state_table['number'].sort_values(inplace=False))

    for sname1, sname2 in itt.combinations(em_table.index.tolist(), r=2):
        # Note
        # the index of the emission table is the state name (by default),
        # hence, need to get the state number here first
        snum1 = state_table.at[sname1, 'number']
        snum2 = state_table.at[sname2, 'number']
        # note to self here:
        # here, it is sufficient to iterate over all
        # state pairs but the identity pairs, since
        # the JSD for the identity pair will be 0 anyway
        state_div = jsd(em_table.loc[sname1, :].values, em_table.loc[sname2, :].values)
        em_score.loc[snum1, snum2] = state_div
        em_score.loc[snum2, snum1] = state_div
    # shift by row mean:
    # turn into notion of similar - neutral - dissimilar
    row_means = em_score.mean(axis=1)
    em_score -= row_means
    # multiply by 10 and round:
    # scores are integers "by convention"; the parameter
    # estimation routines actually implement this in a
    # hard way
    em_score *= 10
    em_score = em_score.round(decimals=0)
    # If option is set:
    # the minimum score is the minimal JSD,
    # and thus the highest similarity between
    # any two state (= what we don't want to find)
    # (after shifting by the row mean).
    # We use this value to punish occurrences of the
    # background state as its lack of signal cannot
    # be interpreted
    penalty = em_score.min(axis=1).min()
    for s1, s2 in itt.combinations(em_score.index.tolist(), r=2):
        # note to self here:
        # can skip over identity pairs, does not change anything
        em_score.loc[s2, s1] = em_score.at[s1, s2]
        if args.bgtreatment == 'penalized':
            if s1 == args.bgstate or s2 == args.bgstate:
                em_score.loc[s1, s2] = penalty
                em_score.loc[s2, s1] = penalty
    if args.bgtreatment == 'penalized':
        em_score.loc[args.bgstate, args.bgstate] = penalty
    # turn rounded floats into integers...
    em_score = em_score.astype(np.int16)
    # confirm that the scoring matrix is indeed symmetric
    assert (em_score == em_score.transpose()).all(axis=1).all(), \
        'Emission scoring matrix not symmetric: {}'.format(em_score)

    if args.bgtreatment == 'penalized':
        store_path = os.path.join(ROOT_SCORE, 'penem')
    else:
        store_path = os.path.join(ROOT_SCORE, 'ordem')
    logger.debug('Storing emission-based scoring matrix '
                 'with {} background state under root: {}'.format(args.bgtreatment, store_path))

    store_scoring_matrix(args.dataset, store_path, em_score, var_lens, args.replace)

    return 0


def store_scoring_matrix(dataset, root_path, score_matrix, var_lens, replace):
    """
    :param dataset:
    :param root_path:
    :param score_matrix:
    :param var_lens:
    :return:
    """
    # load state base frequencies for KA
    # parameter estimation
    with pd.HDFStore(dataset, 'r') as hdf:
        if not replace:
            if any([k.startswith(root_path) for k in hdf.keys()]):
                raise ValueError('Scoring with name "{}" already'
                                 ' exists in dataset'.format(os.path.split(root_path)[-1]))
        basefreq = hdf[PATH_MD_BASEFREQ]
    # estimate parameters per chromosome
    md_columns = ['ka_lambda', 'ka_h', 'ka_k', 'expected_score', 'var_len']
    md_rows = []
    md_index = []
    score_range = None
    for chrom, var_len in var_lens.items():
        chrom_base_freq = basefreq.loc[chrom, :]

        score_range, score_probs = compute_score_probabilities(score_matrix, chrom_base_freq)
        score_expect = np.average(score_range, weights=score_probs)
        assert score_expect < 0, 'Expected score >= 0 for chromosome {}: {}'.format(chrom, score_expect)

        # theoretically, the score_range could start or end
        # with a 0 prob. score, filter for that before
        # estimating parameters
        obs_min = (score_range[score_probs > 0]).min()
        obs_max = (score_range[score_probs > 0]).max()
        nz_indices = np.argwhere(score_probs).flatten()
        limit_probs = score_probs[nz_indices[0]:nz_indices[-1] + 1]

        ka_lambda, ka_h, ka_k = karlin.estimateParameters(obs_min, obs_max, limit_probs)

        md_rows.append([ka_lambda, ka_h, ka_k, score_expect, var_len] + score_probs.tolist())
        md_index.append(chrom)
    score_header = []
    for s in score_range:
        if s < 0:
            score_header.append('score_m{}_prob'.format(-s))
        elif s > 0:
            score_header.append('score_p{}_prob'.format(s))
        else:
            score_header.append('score_0_prob')
    md_columns += score_header
    score_params = pd.DataFrame(md_rows, index=md_index, columns=md_columns,
                                dtype=np.float32)

    with pd.HDFStore(dataset, 'a') as hdf:
        score_mat_path = os.path.join(root_path, 'matrix')
        hdf.put(score_mat_path, score_matrix, format='fixed')
        score_param_out = os.path.join(root_path, 'parameters')
        hdf.put(score_param_out, score_params, format='fixed')

    return None


def set_background_state(args):
    """
    :param args:
    :return:
    """
    if args.bgstate > -1:
        # explicit overwrite requested by user
        pass
    else:
        with pd.HDFStore(args.dataset, 'r') as hdf:
            md_state = hdf[PATH_MD_STATE]
            bgs = int(md_state.loc[md_state['background'] == 1, 'number'][0])
            args.__setattr__('bgstate', bgs)
    return args


def _run_cmd_score(args, logger):
    """
    :param args:
    :return:
    """
    # since this command is supposed to
    # add data to an existing dataset,
    # check that it really does first...
    assert os.path.isfile(args.dataset), 'Path to SCIDDO dataset invalid: {}'.format(args.dataset)

    args = set_background_state(args)
    scorings_to_add = []
    scoring_functions = {'emission': compute_emission_scoring,
                         'replicate': compute_replicate_scoring,
                         'auto': lambda x: False}
    score_names = []
    for entry in args.addscoring:
        if entry == 'auto':
            raise ValueError('The keyword "auto" is used internally and'
                             ' cannot be used to name a scoring schema.')
        comp_fun = scoring_functions.get(entry, None)
        if comp_fun is None:
            assert entry not in scoring_functions, \
                'The following scoring names are used internally' \
                ' and cannot be used to add custom scorings: {}'.format(' - '.join(sorted(scoring_functions.keys())))
            score_names.append(entry)
        else:
            scorings_to_add.append(comp_fun)

    score_files = []
    for fpath in args.scoringfile:
        assert os.path.isfile(fpath), 'Given path to scoring file is invalid: {}'.format(fpath)
        score_files.append(fpath)
    assert len(score_names) == len(score_files) or not score_names, \
        'You have to specify one name per score file or no names at all,' \
        ' otherwise name-to-file mapping is ambiguous.'
    python_var_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]+")
    if not score_names:
        derived_names = []
        for fpath in score_files:
            fname = os.path.basename(fpath)
            scoring_name = fname.rsplit('.', 1)[0]
            assert python_var_re.match(scoring_name) is not None, \
                'Invalid name for scoring scheme detected: {}' \
                ' See "--help" for info about valid names.'.format(scoring_name)
            assert scoring_name not in scoring_functions, \
                'The following scoring names are used internally' \
                ' and cannot be used to add custom scorings: {}'.format(' - '.join(sorted(scoring_functions.keys())))
            derived_names.append(scoring_name)
        score_names = derived_names
    else:
        assert all([python_var_re.match(n) is not None for n in score_names]), \
            'At least one name for the scoring schemes is not valid: {}' \
            ' See "--help" for info about valid names.'.format(score_names)
    assert len(set(score_names)) == len(score_names), \
        'Names of scoring schemes have to be unique: {}'.format(score_names)

    for scf in scorings_to_add:
        logger.debug('Derive scoring from data')
        for handling in args.bgtreatment:
            tmp = cp.deepcopy(args)
            tmp.__setattr__('bgtreatment', handling)
            scf(tmp, logger)
        logger.debug('Done')

    if score_files:
        for fpath, sname in zip(score_files, score_names):
            logger.debug('Reading scoring "{}" from file {}'.format(sname, fpath))
            read_scoring_scheme(args, logger, sname, fpath)

    logger.debug('All requested scores computed - exiting...')
    return 0
