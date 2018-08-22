#!/bin/bash

cd /home/peter/work/repos/github/ptrebert/sciddo

source activate sciddo

python setup.py develop \
 --install-dir /home/peter/tools/miniconda/envs/sciddo/lib/python3.6/site-packages \
 --script-dir /home/peter/tools/miniconda/envs/sciddo/bin \
 --build-directory /home/peter/work/repos/github/ptrebert/sciddo/build

source deactivate