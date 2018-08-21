# coding=utf-8

import os as os
import re as re
import collections as col

import numpy as np
import numpy.random as rng
import pandas as pd

from scidlib.auxiliary import read_key_value_map


def determine_sample_labels(filepaths, label_arg, logger):
    """
    :param filepaths:
    :param label_arg:
    :return:
    """
    design_matrix = None
    python_var_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]+")
    if not label_arg:
        # empty label argument implies: strip extension and be done
        sample_labels = dict()
        for fp in filepaths:
            fn = os.path.basename(fp)
            label = fn.rsplit('.', 1)[0]
            sample_labels[fn] = label
    elif len(label_arg) == 1 and os.path.isfile(label_arg[0]):
        sample_labels = read_key_value_map(label_arg)
    elif len(label_arg) > 1 and label_arg[0] != 'REGEXP':
        # labels are given as list of strings
        sample_labels = dict([(os.path.basename(fp), lab) for fp, lab in zip(filepaths, label_arg)])
    elif len(label_arg) == 2 and label_arg[0] == 'REGEXP':
        # derive labels using regular expression matching
        # on full file paths
        re_str = label_arg[1]
        assert '<LABEL>' in re_str, 'Named group LABEL is required for regular expression labeling'
        re_obj = re.compile(re_str)
        group_values = col.defaultdict(set)
        group_indicators = col.Counter()
        sample_labels = dict()
        for fp in filepaths:
            fn = os.path.basename(fp)
            mobj = re_obj.search(fp)
            if mobj is None:
                logger.warning('Could not match file path {} with regular expression {}'.format(fp, label_arg[1]))
                mobj = re_obj.search(fn)
                if mobj is None:
                    raise ValueError('Could not match file name {} with regular expression {}'.format(fn, label_arg[1]))
                else:
                    logger.debug('Could match file name {} with regular expression'.format(fn))
            label = mobj.group('LABEL')
            assert python_var_re.match(label) is not None, \
                'Label invalid - use only A-Z, a-z, underscore and 0-9 (but not as first character): {}'.format(label)
            sample_labels[fn] = label
            for k, v in mobj.groupdict(default='IGNORE').items():
                if k in ['IGNORE', 'LABEL']:
                    continue
                else:
                    assert python_var_re.match(k) is not None, \
                        'Group name invalid - use only A-Z, a-z, underscore and 0-9' \
                        ' (but not as first character): {}'.format(k)
                    assert python_var_re.match(v) is not None, \
                        'Matched value invalid - can only contain A-Z, a-z, underscore and 0-9' \
                        ' (but not as first character): {}'.format(v)

                    group_indicators[label + '@' + k + '@' + v] += 1
                    group_values[k].add(v)
        design_matrix = build_design_matrix(list(sample_labels.values()),
                                            group_indicators,
                                            group_values)
    else:
        # this must be single file and single label
        assert len(filepaths) == 1 and len(label_arg) == 1, \
            'Expected single file with single label, found: {} and {}'.format(filepaths, label_arg)
        sample_labels = dict([(os.path.basename(filepaths[0]), label_arg[0])])
    assert len(sample_labels) == len(filepaths), \
        'Non-unique mapping between input file paths \
        and sample labels: {} - {}'.format(filepaths, list(sample_labels.values()))
    return sample_labels, design_matrix


def build_design_matrix(labels, grp_ind, grp_values):
    """
    This function largely exists because Scikit-Learn does not
    play well with string categorical variables for one-hot-encoding

    :return:
    """
    num_columns = sum([len(vals) for vals in grp_values.values()])
    num_rows = len(labels)
    row_index = sorted(labels)
    column_index = []
    for group, values in grp_values.items():
        for v in values:
            column_index.append(group + '_' + v)
    column_index = sorted(column_index)
    design_matrix = pd.DataFrame(np.zeros((num_rows, num_columns), dtype=np.int8),
                                 dtype=np.int8, index=row_index, columns=column_index)
    for k, count in grp_ind.items():
        if count > 1:
            raise ValueError('Group membership can only be binary 0-1: {} - {}'.format(k, count))
        row, column_prefix, column_suffix = k.split('@')
        design_matrix.loc[row, column_prefix + '_' + column_suffix] = count
    design_matrix.index.name = 'label'
    return design_matrix


def read_design_matrix(fpath, sample_labels):
    """
    Read design matrix from user specified file.
    If necessary, adjust label column and replace
    filenames with sample labels.
    """
    # sep=None should use csv.Sniffer to infer delimiter
    dm = pd.read_csv(fpath, sep=None, header=0,
                     index_col=0, engine='python')
    dm = dm.astype(np.int8)
    if dm.index.isin(sample_labels).all():
        # design matrix contains file names as sample labels
        dm.index.replace(sample_labels, inplace=True)
        assert dm.index.isin(sample_labels.values()).all(), 'Replacing filename with sample labels failed: {}'.format(dm.index)
        dm.index.name = 'label'
    elif dm.index.isin(sample_labels.values()).all():
        # design matrix contains actual labels
        dm.index.name = 'label'
    else:
        raise ValueError('First column in design matrix must contain filenames or sample labels. \
                          Could not match entries to any of these: {}'.format(dm.head()))
    dm.sort_index(axis=0, inplace=True)
    return dm


def get_replicate_pairs(design_matrix):
    """
    Extract replicated sample pairs (no duplicates) given
    the design matrix

    :param design_matrix:
    :return:
    """
    rep_idx = design_matrix.duplicated(keep=False)
    all_reps = design_matrix.loc[rep_idx, :]
    # identify all equivalent rows
    replicates = all_reps.apply(lambda x: (all_reps == x).all(axis=1), axis=1)
    pairs = []
    seen = set()
    for row in replicates.itertuples():
        p1 = row.Index
        matches = [k for k, v in row._asdict().items() if v and k != p1]
        for p2 in sorted(matches):
            if p2 == 'Index':
                continue
            if not ((p1, p2) in seen or (p2, p1) in seen):
                pairs.append((p1, p2))
                seen.add((p1, p2))
                seen.add((p2, p1))
    return sorted(pairs)


def get_nonreplicate_pairs(design_matrix):
    """
    Extract non-replicated sample pairs (no duplicates) given
    the design matrix. If the design matrix contains replicate groups,
    one sample will be selected at random from each group.

    :param design_matrix:
    :return:
    """
    rep_idx = design_matrix.duplicated(keep=False)
    non_rep_samples = design_matrix.index[~rep_idx].values.tolist()
    all_reps = design_matrix.loc[rep_idx, :]
    # identify all equivalent rows
    replicates = all_reps.apply(lambda x: (all_reps == x).all(axis=1), axis=1)
    rep_groups = dict()
    rep_group_counter = 0
    for row in replicates.itertuples():
        p1 = row.Index
        if p1 not in rep_groups:
            rep_group_counter += 1
            rep_groups[p1] = rep_group_counter
        matches = [k for k, v in row._asdict().items() if v and k != p1]
        for p2 in matches:
            if p2 == 'Index':
                continue
            if p2 not in rep_groups:
                rep_groups[p2] = rep_group_counter
    joined_groups = col.defaultdict(set)
    [joined_groups[gid].add(sample) for sample, gid in rep_groups.items()]
    for _, replicates in joined_groups.items():
        this_sample = rng.choice(list(replicates), 1)[0]
        non_rep_samples.append(this_sample)
    non_rep_samples = sorted(non_rep_samples)
    pairs = []
    seen = set()
    for p1 in non_rep_samples:
        for p2 in non_rep_samples:
            if not ((p1, p2) in seen or (p2, p1) in seen):
                pairs.append((p1, p2))
                seen.add((p1, p2))
                seen.add((p2, p1))
    return sorted(pairs)


def derive_merge_labels(sample_pairs, design_matrix):
    """
    A heuristic to derive group labels used during the merging
    of all high scoring segments pairs resulting from the
    individual replicate comparisons.

    If it is not possible to extract common properties from
    the design matrix, then collate all sample names (could be ugly)

    :param sample_pairs:
    :param design_matrix:
    :return:
    """
    samples1 = sorted(set([p[0] for p in sample_pairs]))
    samples2 = sorted(set([p[1] for p in sample_pairs]))
    if design_matrix is not None:
        sub_g1 = design_matrix.loc[design_matrix.index.isin(samples1), :].all(axis=0)
        sub_g2 = design_matrix.loc[design_matrix.index.isin(samples2), :].all(axis=0)

        sub_g1 = set(sub_g1.index[sub_g1])
        sub_g2 = set(sub_g2.index[sub_g2])

        props_g1 = sorted(sub_g1 - sub_g2)
        props_g2 = sorted(sub_g2 - sub_g1)
        if props_g1 and props_g2:
            mrg_label1 = '-AND-'.join(props_g1)
            mrg_label2 = '-AND-'.join(props_g2)
        else:
            mrg_label1 = '-OR-'.join(samples1)
            mrg_label2 = '-OR-'.join(samples2)
    else:
        mrg_label1 = '-OR-'.join(samples1)
        mrg_label2 = '-OR-'.join(samples2)
    return mrg_label1, mrg_label2
