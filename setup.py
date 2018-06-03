# coding=utf-8

import os as os

from setuptools import setup, find_packages, Extension
import numpy.distutils.misc_util as nputil


def extract_package_info():
    """
    :return:
    """
    setup_path = os.path.dirname(os.path.abspath(__file__))
    pck_path = os.path.join(setup_path, 'src', 'scidlib', '__init__.py')
    assert os.path.isfile(pck_path), \
        'Could not detect SCIDDO-lib init under path {}'.format(os.path.dirname(pck_path))
    infos = dict()
    with open(pck_path, 'r') as init:
        for line in init:
            if line.startswith('__'):
                k, v = line.split('=')
                infos[k.strip()] = eval(v.strip())
    return infos


def read_requirements():
    """
    :return:
    """
    setup_path = os.path.dirname(os.path.abspath(__file__))
    req_path = os.path.join(setup_path, 'requirements.txt')
    req_pck = []
    with open(req_path, 'r') as req:
        for line in req:
            if line.strip():
                req_pck.append(line.strip())
    return req_pck


def locate_executable():
    """
    :return:
    """
    setup_path = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(setup_path, 'bin', 'sciddo.py')
    relpath_script = os.path.relpath(script_path)
    return relpath_script


def locate_karlin_source():
    """
    :return:
    """
    setup_path = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(setup_path, 'src', 'scidlib', 'karlinmodule.c')
    relpath_script = os.path.relpath(script_path)
    return relpath_script


karlin_module = Extension(name='karlin',
                          sources=[locate_karlin_source()],
                          extra_compile_args=["-Wno-deprecated"],
                          include_dirs=nputil.get_numpy_include_dirs(),
                          optional=False)

pck_infos = extract_package_info()

setup(
    name="scidlib",
    version=pck_infos['__version__'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # removed this - no longer necessary
    # since Karlin module was turned into
    # a proper C extension
    #package_data={'scidlib': ['karlin.so']},
    scripts=[locate_executable()],
    install_requires=read_requirements(),
    ext_modules=[karlin_module],

    author=pck_infos['__author__'],
    author_email=pck_infos['__email__'],
    description="Score-based identification of differential chromatin regions",
    license="GPLv3",
    url="https://github.com/ptrebert/sciddo"   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
