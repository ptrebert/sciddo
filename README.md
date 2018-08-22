# SCIDDO: Sciddo-based identification of differential chromatin domains

## Publication
in preparation

## Code maturity
SCIDDO is currently in BETA status

## Setup
SCIDDO supports only Linux environments (that is unlikely to change in the future) and is developed using Python3.6.
Other Python3.x versions may or may not work, but are not officially supported.

For easy setup, it is highly recommended to install SCIDDO inside a dedicated Conda environment.
A suitable environment is specified in `environments/sciddo_env.yml`.

Otherwise, install the HDF5 library (tested with version 1.8.18) as appropriate for your local environment,
and the necessary Python dependencies from the `requirements.txt` file:

```bash
sudo apt-get install libhdf5\
sudo pip install -r requirements.txt
```

Empirically, the setup of PyTables and HDF5 can create some headaches.
In this case, the best advice is to use Conda.

After all dependencies have been installed successfully,
run the SCIDDO setup as appropriate for your environment:

```bash
[sudo] python setup.py install
```

## Execution

### Getting help

`sciddo.py --help` or `sciddo.py <SUBCOMMAND> --help` is your friend.

### Standard analysis run

A standard SCIDDO analysis run is split into several distinct steps that are realized by different code modules.
Besides module specific parameters, there are several global parameters to adjust SCIDDO's runtime behavior.
Importantly, these global parameters always have to be specified before the subcommand, i.e.,

```
sciddo.py [GLOBAL_PARAMETERS] <SUBCOMMAND> [MODULE_PARAMETERS]
```

The global parameters are:

```bash
--workers: number of CPUs to use (no sanity checks!)\
--debug: print debug messages to stderr; otherwise, SCIDDO operates silently\
--config-dump: folder to dump run configuration (JSON); defaults to current working directory\
--no-dump: do not dump run configuration
```

#### Step 1: convert
 
Convert all input data (state segmentations plus metadata) into a binary HDF5 file. Currently, ChromHMM
and EpiCSeg output files are supported out-of-the-box. This creates the SCIDDO DATA file.

```bash
sciddo.py [GLOBAL_PARAMETERS] convert --help
```

#### Step 2: stats

Compute a bunch of statistics (e.g., state composition per sample) that are potentially needed downstream.

```bash
sciddo.py [GLOBAL_PARAMETERS] stats --help
```

#### Step 3: score

Add scoring schemes (matrices) to the dataset. These can be derived automatically from the state segmentation
model emissions (if provided during the convert step), or can be supplied in form of a user-defined file.
Note that, in principle, an arbitrary number of scoring schemes can be added to a dataset.

```bash
sciddo.py [GLOBAL_PARAMETERS] score --help
```

#### Step 4: scan

Scan the dataset for differential chromatin domains. As opposed to the previous commands, this creates a separate
output file per run, i.e., the SCIDDO RUN file.

```bash
sciddo.py [GLOBAL_PARAMETERS] scan --help
```

#### Step 5: dump

All data and metadata in the SCIDDO DATA and RUN file can be dumped to text files (e.g., TSV tables or BED files) for downstream analysis.

```bash
sciddo.py [GLOBAL_PARAMETERS] dump --help
```