# SCIDDO: Sciddo-based identification of differential chromatin domains

## Publication
in preparation

## Setup
SCIDDO supports only Linux environments (that is unlikely to change in the future) and is developed using Python3.6.
Other Python3.x versions may or may not work, but are not officially supported.

For easy setup, it is highly recommended to run SCIDDO inside a dedicated Conda environment. A suitable environment is specified in `environments/sciddo_env.yml`.
Otherwise, install the HDF5 library (tested with version 1.8.4) as appropriate for your local environment, e.g.,

```
sudo apt-get install libhdf5-8
```

and then run the Python setup script as appropriate for your environment

```
[sudo] python setup.py install
```

## Execution
