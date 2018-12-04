# coding=utf-8

# package info
__author__ = 'Peter Ebert'
__email__ = 'pebert@mpi-inf.mpg.de'
<<<<<<< HEAD
__version__ = '0.8.0'
=======
__version__ = '0.7.1'
>>>>>>> origin/develop


# some formatting config
LOG_FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
TIME_FORMAT = '%Y%m%d-%H%M%S'

# artifical order numbers for non-numeric chromosome names
CHROM_ORDER = {'2A': 100, '2B': 101,
               'X': 102, 'Y': 103, 'M': 104,
               'MT': 105, 'W': 106, 'Z': 107}

# Output columns of BED file when limiting output
BED_LIMIT = '#chrom chromStart chromEnd name nat_score segment_expect segment_pv nat_score_lnorm'

# numeric precision/accuracy for p-value computation
NUM_PREC_KA_PV = 2**10
# size restriction on score int
# note that this is a string
# to avoid importing modules here
NUM_MAX_SIZE_SCORE = 'np.int16'


# Some default values for paths in SCIDDO datasets
# PATH = full path
# ROOT = path prefix

# for metadata
PATH_MD_INPUT = '/metadata/inputs'
PATH_MD_EMISSION = '/metadata/emissions'
PATH_MD_DESIGN = '/metadata/design'
PATH_MD_STATE = '/metadata/states'
PATH_MD_CHROM = '/metadata/chromosomes'
PATH_MD_COMPARISON = '/metadata/comparisons'
PATH_MD_COMP_PARAMS = '/metadata/parameters'
PATH_MD_BASEFREQ = '/metadata/basefreq'
ROOT_MD_TRANS = '/metadata/transitions'
PATH_MD_TRANS_REP = ROOT_MD_TRANS + '/replicates'
PATH_MD_TRANS_SNG = ROOT_MD_TRANS + '/singletons'

# for counting statistics
ROOT_STAT = '/statistic'
ROOT_STAT_COUNT_COMP = ROOT_STAT + '/counts/composition'
ROOT_STAT_COUNT_TRANS = ROOT_STAT + '/counts/transitions'
ROOT_STAT_COUNT_TRANS_REP = ROOT_STAT_COUNT_TRANS + '/replicates'
ROOT_STAT_COUNT_TRANS_SNG = ROOT_STAT_COUNT_TRANS + '/singletons'
ROOT_STAT_AGREE_COUNT = ROOT_STAT + '/agreement/raw'
ROOT_STAT_AGREE_SCORE = ROOT_STAT + '/agreement/score'


# for scoring schemes
ROOT_SCORE = '/scoring'

# for results
ROOT_SCAN = '/raw_scan'
ROOT_SAMPLE = '/raw_sample'
ROOT_HSP = '/segments'
