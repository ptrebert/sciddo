#!/bin/bash

cd /home/pebert/work/code/github/sciddo

source activate sciddo

python setup.py develop \
 --install-dir /TL/epigenetics2/work/pebert/conda/envs/sciddo/lib/python3.6/site-packages \
 --script-dir /TL/epigenetics2/work/pebert/conda/envs/sciddo/bin \
 --build-directory /home/pebert/work/code/github/sciddo/build

source deactivate