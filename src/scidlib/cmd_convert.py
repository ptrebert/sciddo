# coding=utf-8

import os as os
import collections as col
import fnmatch as fnm
import multiprocessing as mp

import pandas as pd

from scidlib import PATH_MD_INPUT, PATH_MD_STATE, \
    PATH_MD_CHROM, PATH_MD_EMISSION, PATH_MD_DESIGN
from scidlib.data_converters import get_file_processor, get_emission_parser
from scidlib.auxiliary import collect_files, read_key_value_map, read_chrom_sizes
from scidlib.sample_handling import determine_sample_labels, read_design_matrix


def add_convert_cmd_parser(subparsers):
    """
    :param subparsers:
    :return:
    """
    parser_conv = subparsers.add_parser('convert',
                                        help='Convert state segmentation data plus annotation files'
                                             ' into a SCIDDO dataset.')

    grp = parser_conv.add_argument_group('Input paths')
    grp.add_argument('--state-seg', '-ssg', type=str, nargs='+', dest='stateseg', required=True,
                     help='Path to input folder with state segmentation BED files or'
                          ' space-separated full paths to several segmentation files.')
    grp.add_argument('--state-post', '-spo', type=str, nargs='*', dest='statepost', default=[],
                     help='Path to input folder with state posterior text files or'
                          ' space-separated full paths to several posterior files.'
                          ' Default: <empty>')
    grp.add_argument('--chrom-sizes', '-csz', type=str, dest='chromsizes', required=True,
                     help='Full path to two column text file stating chromosome'
                          ' names and sizes.')
    grp.add_argument('--state-labels', '-stl', type=str, dest='statelabels', default='IGNORE',
                     help='Full path to two column text file stating more descriptive'
                          ' names for the individual chromatin states. Note: the special'
                          ' value EXTRACT will trigger an on-the-fly extraction of the name -'
                          ' in this case, the name column in the input BED file must conform'
                          ' to this naming style: STATE-NUM_STATE-LABEL. These names will be split'
                          ' into (STATE-NUM, STATE-LABEL). Default: IGNORE')
    grp.add_argument('--state-colors', '-stc', type=str, dest='statecolors', default='0',
                     help='Full path to two column text file stating RGB colors'
                          ' for the individual chromatin states. Note: if you specify'
                          ' an integer, this will trigger an on-the-fly extraction'
                          ' of the RGB color from the respective column of the state segmentation'
                          ' files (1-based, i.e., first column is 1, n-th column is N).'
                          ' The value "9" is valid for files strictly adhering to the BED'
                          ' format as specified here:'
                          ' https://genome.ucsc.edu/FAQ/FAQformat.html --> BED format // itemRgb column'
                          ' A value of 0 skips the annotation with state colors.'
                          ' Default: 0')
    grp.add_argument('--design-matrix', '-dm', type=str, dest='designmatrix', default='',
                     help='Full path to design matrix of the experiments. This must be an'
                          ' indicator matrix with header: first column has to be the sample'
                          ' label (see below) or the full file name (w/o folder). All remaining'
                          ' columns are assumed to represent group characteristics such as sex,'
                          ' cell, tissue type or treatment conditions in one-hot-encoding.'
                          ' Example: for sex, two columns should be present (with header!),'
                          ' with a single 1 per sample indicating whether or not it is a male'
                          ' or a female sample; the respective other column has to be 0.'
                          ' Default: <empty>')
    grp.add_argument('--model-emissions', '-mde', type=str, dest='modelemissions', default='',
                     help='Full path to table of model emission probabilities. The format'
                          ' of the table must match the segmentation format as specified below'
                          ' (see option --seg-format).'
                          ' If specified, model emission probabilities can be used'
                          ' to derive a scoring scheme for state similarities. Default: <empty>')

    grp = parser_conv.add_argument_group('Input parameters')
    grp.add_argument('--bin-size', '-bin', type=int, default=200, dest='binsize',
                     help='Bin size of the state segmentation. The minimum bin size'
                          ' currently supported is 100bp. Default: 200')

    grp.add_argument('--seg-format', '-sfm', type=str, choices=['ChromHMM', 'EpiCSeg'],
                     default='ChromHMM', dest='segformat',
                     help='Specify the input format depending on the tool'
                          ' that generated the state segmentation. Default: ChromHMM')
    grp.add_argument('--emission-format', '-emf', type=str, choices=['generic', 'model'],
                     default='model', dest='emissionformat',
                     help='Specify the format of the state emission matrix. "generic" requires'
                          ' a comma or tab-separated table with (hidden) state numbers as'
                          ' rows and observed data (e.g., histone marks) as columns. "model"'
                          ' implies that you submit the model file (following the value'
                          ' set for "--seg-format") itself and the emission probabilities'
                          ' are to be extracted from this file. Default: model')

    grp.add_argument('--sample-labels', '-slb', type=str, nargs='*', default=[], dest='samplelabels',
                     help='Specify a (short) sample label per input file as a space-separated'
                          ' list - note that order of labels must match order of state segmentation'
                          ' files. If left empty, this implies to strip the file extension'
                          ' (usually .bed) and use the remainder of the file name as label.'
                          ' The special sequence " --sample-labels REGEXP "CUSTOM_REGEXP" "'
                          ' triggers regular expression based generation of sample labels.'
                          ' Note that the named group LABEL has to be present in the regular'
                          ' expression. All other named groups will be used to create a design'
                          ' matrix for the samples (see above). If a dedicated design matrix'
                          ' is specified above, this will take precedence over the dynamically'
                          ' derived one here, but deriving sample labels using a regular expression'
                          ' is still possible. For information how to specify regular expressions'
                          ' with named groups in Python, refer to:'
                          ' https://docs.python.org/3/library/re.html'
                          ' Default: <empty>')

    grp.add_argument('--chrom-filter', '-cfl', type=str, default="(chr)?[0-9XY]+(\s|$)", dest='chromfilter',
                     help='Specify regular expression as raw string (i.e.: "REGEXP") to'
                          ' filter chromosome based on name. Only the reduced set of'
                          ' chromosomes will be stored in the output. The default expression'
                          ' filters for auto- and gonosomes in species like human and mouse.'
                          ' Default: "(chr)?[0-9XY]+(\s|$)"')

    grp.add_argument('--collect-input', '-col', action='store_true', default=False, dest='collectinput',
                     help='If set, collect input files (both segmentation and posterior files) by'
                          ' recursively collecting files from all sub folders. This is ignored if the'
                          ' input arguments are a list of files instead of folder paths. Default: False')

    grp = parser_conv.add_argument_group('Output path')
    grp.add_argument('--output', '-o', type=str, required=True,
                     help='Full path to output file. Folder(s) will be created if necessary.')

    parser_conv.set_defaults(execute=_run_cmd_convert)
    return subparsers


def _standard_file_patterns(tool):
    """
    Store default file extensions produced
    by segmentation tools. As long as users
    do not change the file extensions, SCIDDO
    should be able to process known segmentations

    :param tool:
    :return:
    """
    seg = {'ChromHMM': '*_segments.bed',
           'EpiCSeg': '*.bed'}
    post = {'ChromHMM': '*_posterior.txt',
            'EpiCSeg': '*'}
    return seg[tool], post[tool]


def _standard_file_split(tool):
    """
    :param tool:
    :return:
    """
    seg = {'ChromHMM': 2, 'EpiCSeg': 1}
    # EpiCSeg does not produce posterior
    # files by default
    post = {'ChromHMM': 3, 'EpiCSeg': 0}
    return seg[tool], post[tool]


def prepare_posterior_files(sample_posteriors, sample_label, chromosomes, binsize, segformat):
    """
    Annotate each posterior file with the respective sample label. This function relies on
    standard naming schema based on the user-specified segmentation format / tool.
    """
    params = []
    for fp in sample_posteriors:
        if segformat == 'ChromHMM':
            fn = os.path.basename(fp)
            chrom = fn.split('_')[-2]
            if chrom not in chromosomes:
                continue
            param_set = 'posterior', sample_label, chrom, chromosomes[chrom], binsize, fp
            params.append(param_set)
        else:
            raise NotImplementedError('No support for data format: {}'.format(segformat))
    return params


def prepare_input_files(state_files, sample_labels, posteriors, chromosomes,
                        binsize, state_labels, state_colors, segformat):
    """
    :param state_files:
    :param sample_labels:
    :param posteriors:
    :param chromosomes:
    :param binsize:
    :param state_labels:
    :param state_colors:
    :param segformat:
    :return:
    """
    input_params = []
    splits_state, splits_post = _standard_file_split(segformat)
    for fp in state_files:
        fn = os.path.basename(fp)
        label = sample_labels[fn]
        if segformat == 'ChromHMM':
            prefix = fn.rsplit('_', splits_state)[0]
            sample_post = fnm.filter(posteriors, '*' + prefix + '*')
            if posteriors:
                assert sample_post, 'Could not identify sample posteriors: {} - {}'.format(prefix, label)
            post_params = prepare_posterior_files(sample_post, label, chromosomes, binsize, segformat)
            input_params.extend(post_params)
        for chrom, size in chromosomes.items():
            param_set = 'state', label, chrom, size, binsize, fp, state_labels, state_colors
            input_params.append(param_set)
    return input_params


def prepare_inputs_table(inputlist):
    """
    Save basic data about input files
    """
    post_count = col.Counter()
    chrom_count = col.Counter()
    input_files = dict()
    for param_set in inputlist:
        label, fp = param_set[1], param_set[5]
        fn = os.path.basename(fp)
        input_files[label] = fn
        if param_set[0] == 'state':
            chrom_count[label] += 1
        else:
            post_count[label] += 1
    inputs = []
    for label, filename in input_files.items():
        this_file = dict()
        this_file['label'] = label
        this_file['filename'] = filename
        this_file['chrom_count'] = chrom_count[label]
        this_file['posterior_files'] = post_count[label]
        inputs.append(this_file)
    df = pd.DataFrame(inputs)
    df.set_index('label', inplace=True)
    df['chrom_count'] = df['chrom_count'].astype('int32')
    df['posterior_files'] = df['posterior_files'].astype('int32')
    df.index.name = 'label'
    return df   


def prepare_chrom_table(chromosomes, binsize):
    """
    Create Pandas dataframe for chromosomes
    """
    chrom_table = []
    for cname, csize in chromosomes.items():
        this_chrom = dict()
        number_bins = csize // binsize
        last_start = number_bins * binsize - binsize
        this_chrom['name'] = cname
        this_chrom['size_bp'] = csize
        this_chrom['binsize'] = binsize
        this_chrom['bins'] = number_bins
        this_chrom['last_start'] = last_start
        chrom_table.append(this_chrom)
    df = pd.DataFrame(chrom_table)
    df.set_index('name', inplace=True)
    df = df.astype('uint32')

    return df
        

def estimate_background_state(bgcounts):
    """
    :param bgcounts:
    :return:
    """
    total_counts = float(sum([v for v in bgcounts.values()]))
    bg_state, state_count = bgcounts.most_common(1)[0]
    return bg_state, round((state_count / total_counts) * 100, 1)


def prepare_state_table(state_numbers, state_labels, state_colors,
                        fpath_labels, fpath_colors, background):
    """
    Minimalistic annotation for chromatin states
    """
    force_label = False
    force_color = False
    if os.path.isfile(fpath_labels):
        state_labels = read_key_value_map(fpath_labels)
        # there is a label file, so the user
        # expects the states to be labeled, else error
        force_label = True
    if os.path.isfile(fpath_colors):
        state_colors = read_key_value_map(fpath_colors)
        force_color = True
    states = []
    bg_state, bg_freq = estimate_background_state(background)
    for name, number in state_numbers.items():
        this_state = dict()
        this_state['name'] = name
        this_state['number'] = number
        if number == bg_state:
            this_state['background'] = 1
        else:
            this_state['background'] = 0
        try:
            label = state_labels[name]
        except KeyError:
            # could be that the label map uses
            # state numbers instead of names
            label = state_labels.get(str(number), None)
            if label is None and force_label:
                raise KeyError('Could not find label for state {} / {}'.format(name, number))
        try:
            color = state_colors[name]
        except KeyError:
            color = state_colors.get(str(number), None)
            if color is None and force_color:
                raise KeyError('Could not find color for state {} / {}'.format(name, number))
        if label is None:
            this_state['description'] = 'no_description'
        else:
            this_state['description'] = label
        if color is None:
            this_state['color'] = 'no_color'
        else:
            this_state['color'] = color
        states.append(this_state)
    df = pd.DataFrame(states)
    df.set_index('name', inplace=True)
    df.sort_values('number', axis=0, inplace=True)
    return df, bg_state, bg_freq


def _run_cmd_convert(args, logger):
    """
    :param args:
    :return:
    """
    chromosomes = read_chrom_sizes(args.chromsizes, args.chromfilter)
    logger.debug('Read chromosomes: {}'.format(chromosomes))
    if os.path.isdir(args.stateseg[0]):
        logger.debug('Input detected as directory, collecting files...')
        state_files = collect_files(args.stateseg[0], args.collectinput,
                                    _standard_file_patterns(args.segformat)[0])
    else:
        state_files = args.stateseg
    logger.debug('Detected {} state segmentation files as input'.format(len(state_files)))
    labeled_samples, design_matrix = determine_sample_labels(state_files, args.samplelabels, logger)
    if args.designmatrix and os.path.isfile(args.designmatrix):
        logger.debug('Reading design matrix from file {}'.format(args.designmatrix))
        design_matrix = read_design_matrix(args.designmatrix, labeled_samples)
    if args.modelemissions:
        logger.debug('Reading state emission probabilities from file {}'.format(args.modelemissions))
        model_em = get_emission_parser(args.emissionformat, args.segformat)(args.modelemissions)
    else:
        model_em = None
    if not args.statepost:
        post_files = []
    else:
        # processing posterior files could/should be deprecated
        # the information is currently not used
        if os.path.isdir(args.statepost[0]):
            post_files = collect_files(args.statepost[0], args.collectinput,
                                       _standard_file_patterns(args.segformat)[1])
        else:
            post_files = args.statepost
    input_params = prepare_input_files(state_files, labeled_samples, post_files, chromosomes,
                                       args.binsize, args.statelabels, args.statecolors, args.segformat)
    logger.debug('Creating output folder')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    process_files = get_file_processor(args.segformat)
    total_jobs = len(input_params)
    logger.debug('Start processing files - {} jobs in total'.format(total_jobs))
    assert total_jobs > 0, 'No processing jobs created - something went wrong'
    bg_state = col.Counter()
    with pd.HDFStore(args.output, mode='w', complevel=9, complib='blosc') as hdf:
        state_labels = dict()
        state_colors = dict()
        state_numbers = dict()
        job_count = 0
        with mp.Pool(args.workers) as pool:
            resit = pool.imap_unordered(process_files, input_params)
            for data_type, data_path, data_obj, state_info in resit:
                job_count += 1
                logger.debug('Finished job {} of {}: {}'.format(job_count, total_jobs, data_path))
                hdf.put(data_path, data_obj, format='fixed')
                if data_type == 'state':
                    state_numbers.update(state_info[0])
                    state_labels.update(state_info[1])
                    state_colors.update(state_info[2])
                    # select first 50 entries from each
                    # state file to estimate the background
                    # state from the data - majority voting
                    bg_state.update(data_obj[:50].values)
        logger.debug('Saving metadata to output file')
        if design_matrix is not None:
            hdf.put(PATH_MD_DESIGN, design_matrix, format='fixed')
        if model_em is not None:
            em_states = sorted(model_em.index.tolist())
            rd_states = sorted(state_numbers.values())
            assert rd_states == em_states, 'State numbers from emission matrix are different' \
                                           ' from state numbers read from segmentation' \
                                           ' files: {} - {}'.format(em_states, rd_states)
            # now that it is confirmed that states are identical,
            # replace emission index (state numbers) by state names
            # to make its semantics more obvious
            inv_state_num = dict((v, k) for k, v in state_numbers.items())
            new_index = [inv_state_num[i] for i in model_em.index.tolist()]
            model_em.index = new_index
            model_em.index.name = 'state'
            hdf.put(PATH_MD_EMISSION, model_em, format='fixed')

        chrom_df = prepare_chrom_table(chromosomes, args.binsize)
        hdf.put(PATH_MD_CHROM, chrom_df, format='fixed')
        state_df, bg_state, bg_freq = prepare_state_table(state_numbers, state_labels, state_colors,
                                                          args.statelabels, args.statecolors, bg_state)
        logger.debug('Estimated background state: {} ({}%)'.format(bg_state, bg_freq))
        hdf.put(PATH_MD_STATE, state_df, format='table')
        inputs_df = prepare_inputs_table(input_params)
        hdf.put(PATH_MD_INPUT, inputs_df, format='table')
        logger.debug('All metadata saved')
    logger.debug('Output file closed - exiting...')
    return 0
