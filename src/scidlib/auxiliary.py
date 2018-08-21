# coding=utf-8

import os as os
import fnmatch as fnm
import gzip as gz
import bz2 as bz2
import re as re


def open_mode_heuristic(filepath, read=True):
    """
    :param filepath:
    :param read:
    :return:
    """
    if filepath.endswith('.gz') or filepath.endswith('.gzip'):
        if read:
            open_fd, mode = gz.open, 'rt'
        else:
            open_fd, mode = gz.open, 'wt'
    elif filepath.endswith('.bz2') or filepath.endswith('.bzip2'):
        if read:
            open_fd, mode = bz2.open, 'rt'
        else:
            open_fd, mode = bz2.open, 'wt'
    else:
        if read:
            open_fd, mode = open, 'r'
        else:
            open_fd, mode = open, 'w'
    return open_fd, mode


def collect_files(toplevel, recursive=False, pattern='*'):
    """
    :param toplevel:
    :param recursive:
    :param pattern:
    :return:
    """
    if not recursive:
        all_files = [os.path.join(toplevel, fn) for fn in os.listdir(toplevel)]
    else:
        all_files = []
        for root, dirs, files in os.walk(toplevel, followlinks=False):
            if files:
                all_files.extend([os.path.join(root, fn) for fn in files])
    filtered_files = fnm.filter(all_files, pattern)
    return sorted(filtered_files)


def read_key_value_map(fpath):
    """
    :param fpath:
    :return:
    """
    fopen, fmode = open_mode_heuristic(fpath, True)
    with fopen(fpath, fmode) as mapping:
        kv_map = dict([(line.strip().split()) for line in mapping if line.strip()])
    return kv_map


# ========================================
# the following functions are less generic
# but don't fit into another module
# ========================================


def read_chrom_sizes(fpath, chrom_re=None):
    """
    :param fpath:
    :param chrom_re:
    :return:
    """
    if chrom_re is not None:
        chrom_select = re.compile(chrom_re)
    else:
        # whitespace as part of chromosome name is
        # not supported - no discussion...
        chrom_select = re.compile(r"[\w\-\.]+")
    fopen, fmode = open_mode_heuristic(fpath, True)
    chromosomes = dict()
    with fopen(fpath, fmode) as infile:
        for line in infile:
            if line.strip():
                parts = line.strip().split()
                cname = parts[0]
                m = chrom_select.match(cname)
                if m is None:
                    continue
                csize = int(parts[1])
                chromosomes[cname] = csize
    assert chromosomes, 'No chromosome sizes read from file {}'.format(fpath)
    return chromosomes


def sort_columns_bed_order(column_names):
    """
    :param column_names:
    :return:
    """
    lexorder = [(idx, name) for idx, name in enumerate(sorted(column_names), start=100)]
    reorder = []
    for pos, name in lexorder:
        if name in ['chrom', 'chromosome', '#chrom']:
            reorder.append((0, name))
        elif name in ['start', 'start_bp', 'chromStart']:
            reorder.append((1, name))
        elif name in ['end', 'end_bp', 'chromEnd']:
            reorder.append((2, name))
        elif name in ['name', 'ID']:
            reorder.append((3, name))
        elif name in ['score', 'nat_score']:
            reorder.append((4, name))
        elif name in ['strand']:
            reorder.append((5, name))
        elif name in ['nat_score_std']:
            reorder.append((6, name))
        elif name in ['pValue', 'segment_pv']:
            reorder.append((7, name))
        elif name in ['Evalue', 'segment_expect']:
            reorder.append((8, name))
        elif name in ['start_bin']:
            reorder.append((10, name))
        elif name in ['end_bin']:
            reorder.append((11, name))
        else:
            reorder.append((pos, name))
    reorder = sorted(reorder)
    return [t[1] for t in reorder]
