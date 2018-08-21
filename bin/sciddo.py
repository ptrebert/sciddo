#!/usr/bin/env python
# coding=utf-8

import os as os
import sys as sys
import time as ti
import json as js
import random as rnd
import traceback as trb
import argparse as argp
import logging as log
import string as string
import multiprocessing as mp

# Load SCIDDO lib sub commands
from scidlib import LOG_FORMAT, TIME_FORMAT, __version__
from scidlib.cmd_convert import add_convert_cmd_parser
from scidlib.cmd_stats import add_stats_cmd_parser
from scidlib.cmd_score import add_score_cmd_parser
from scidlib.cmd_scan import add_scan_cmd_parser
from scidlib.cmd_dump import add_dump_cmd_parser
# from sciddo.cmd_sample import add_sample_cmd_parser


def parse_command_line():
    """
    :return:
    """
    parser = argp.ArgumentParser(add_help=True)
    parser.add_argument('--version', '-v', action='version', version=__version__)

    comgroup = parser.add_argument_group('General parameters')
    comgroup.add_argument('--workers', '-wrk', type=int, default=1, dest='workers',
                          help='Number of CPU cores to use. This value will be used'
                               ' w/o additional plausibility checks, so be sure that'
                               ' the specified number of CPU cores is actually available'
                               ' on your machine. Default: 1')
    parser.add_argument('--debug', '-dbg', action='store_true', default=False, dest='debug',
                        help='Print status/progress messages on stderr. Otherwise, SCIDDO'
                             ' will be silent and will only report errors on stderr.')

    parser.add_argument('--config-dump', '-conf', type=str, default=os.getcwd(), dest='confdump',
                        help='Specify a folder to save a run configuration file'
                             ' after a successful SCIDDO run. If "--debug"'
                             ' is set, the configuration file will be dumped'
                             ' before the run starts and updated after a'
                             ' successful run. Default: {}'.format(os.getcwd()))

    parser.add_argument('--no-conf-dump', '-nod', action='store_true', default=False, dest='nodump',
                        help='Do not store a run configuation file. Default: False')

    parser = add_sub_parsers(parser)

    return parser.parse_args()


def get_env_copy():
    """
    :return:
    """
    run_env = os.environ.copy()
    use_keys = ['home', 'path', 'pythonpath', 'user', 'hostname', 'shell',
                'homebin', 'logname', 'ssh_connection', 'ld_library_path', 'sty']
    limit_env = dict()
    for k, v in run_env.items():
        lut_key = k.lower()
        if lut_key in use_keys:
            limit_env[k] = v
        elif 'conda' in lut_key and 'backup' not in lut_key:
            limit_env[k] = v
        else:
            pass
    return limit_env


def store_run_config(args, logger, start_time, end_time, start_date, update_path):
    """
    :param args:
    :param logger:
    :param start_time:
    :param end_time:
    :param start_date:
    :param update_path: if not None, update existing configuration dump
    :return:
    """
    if update_path is not None:
        fpath = update_path
    else:
        os.makedirs(args.confdump, exist_ok=True)
        rand_id = ''.join(rnd.choices(string.ascii_lowercase, k=4))
        timestamp = ti.strftime(TIME_FORMAT)
        fname = timestamp + '-{}_SCIDDO_{}_cfg.json'.format(rand_id, args.subparser_name)
        fpath = os.path.join(args.confdump, fname)
        assert not os.path.isfile(fpath), 'Config dump path already exists: {}'.format(fpath)
    run_env = get_env_copy()
    run_conf = dict(vars(args))
    run_conf['execute'] = run_conf['execute'].__name__
    if end_time > 0:
        run_time = str(round((end_time - start_time) / 60, 1))
        end = ti.ctime(end_time)
    else:
        run_time = 'unfinished'
        end = 'unfinished'
    if start_date is None:
        run_date = ti.ctime()
    else:
        run_date = start_date
    run_info = {'env': run_env, 'args': run_conf, 'date': run_date,
                'start': ti.ctime(start_time), 'end': end, 'time_min': run_time,
                'sciddo_version': __version__}
    with open(fpath, 'w') as conf:
        js.dump(run_info, conf, ensure_ascii=True, indent=1)
    logger.debug('Configuration written to {}'.format(fpath))
    return fpath


def add_sub_parsers(main_parser):
    """
    :param main_parser:
    :return:
    """
    subparsers = main_parser.add_subparsers(dest='subparser_name', title='Run modes')
    subparsers = add_convert_cmd_parser(subparsers)
    subparsers = add_stats_cmd_parser(subparsers)
    subparsers = add_score_cmd_parser(subparsers)
    subparsers = add_scan_cmd_parser(subparsers)
    subparsers = add_dump_cmd_parser(subparsers)
    # subparsers = add_sample_cmd_parser(subparsers)
    return main_parser


if __name__ == '__main__':
    exc = 0
    start_time = int(ti.time())
    start_date = ti.ctime()
    conf_path = None
    logger = None
    rnd.seed()
    mp.set_start_method('forkserver')
    try:
        cli_args = parse_command_line()
        if cli_args.debug:
            log.basicConfig(**{'stream': sys.stderr, 'level': log.DEBUG, 'format': LOG_FORMAT})
        else:
            log.basicConfig(**{'stream': sys.stderr, 'level': log.WARNING, 'format': LOG_FORMAT})
        logger = log.getLogger()
        if cli_args.debug and not cli_args.nodump:
            conf_path = store_run_config(cli_args, logger, start_time, 0, start_date, conf_path)
        exc = cli_args.execute(cli_args, logger)
        end_time = int(ti.time())
        if not cli_args.nodump:
            _ = store_run_config(cli_args, logger, start_time, end_time, start_date, conf_path)
    except Exception as e:
        trb.print_exc()
        if logger is not None:
            logger.error('Properly handled shutdown upon error:\n\n{}'.format(e))
            if conf_path is not None:
                logger.debug('Run configuration stored at: {}'.format(conf_path))
        exc = 1
    finally:
        log.shutdown()
        sys.exit(exc)
