## SCIDDO tutorials

The following step-by-step instructions showcase how to use SCIDDO on a data set to identify
differential chromatin domains (DCDs). The test data are an excerpt of a real world data set
(see below for full references) to keep the data volume low enough such that this tutorial
can be followed on any reasonably modern laptop.

**Please note**: this test data set has been compiled to serve as proof of function for SCIDDO.
Following the steps in this tutorial and analyzing the test data does not yield any interesting
biological insights. The goal of this tutorial is to illustrate the necessary commands to go from
a set of chromatin state segmentation files to a list of (candidate) regions of differential
chromatin marking.

#### Test data sources
The original sources for the test data have been published as part of the
[ROADMAP Epigenomics consortium](www.roadmapepigenomics.org)
(see [Nature volume 518, pages317–330, 2015](https://doi.org/10.1038/nature14248])).
We selected four ChromHMM chromatin state segmentation files from two different cellular lineages
(E014 and E016 [embryonic stem cells], E115 and E124 [blood]), and restricted the state segmentation to
chromosome 10 to lower compute requirements (see [testdata/segmentation](testdata/segmentation)).
The ChromHMM model used to generate the state segmentations is the so-called expanded 18-state model,
which is based on six histone marks plus the Input control and was trained on 98 epigenomes.

Besides the chromatin state segmentation files, we downloaded the appropriate ChromHMM model file,
plus the auxiliary files for chromatin state colors and labels. All orignal source files can be
found on the ROADMAP Epigenomics data portal:

[ROADMAP expanded 18-state model data resource](https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/core_K27ac/jointModel/final/)


### Step 0: getting help
The commands explained below usually offer different ways of achieving the same goal, e.g., reading
chromatin state labels from a separate file or extracting them on-the-fly from the chromatin
state segmentation (BED) files. Hence, this tutorial cannot cover all possible use cases, and in order
to check if options better suited to the data at hand are available, SCIDDO's command line help
should be checked:

```
sciddo.py --help  # global help
sciddo.py [run mode] --help  # help specific for selected run mode
```

## Part 1: running a SCIDDO analysis on the test data set

### Step 1: creating a SCIDDO data set
The first step in a SCIDDO analysis is to create a SCIDDO data set which stores the information
of the input data (state segmentations, design matrix etc.) in a more efficient way. The only file we have
to create manually before is the design matrix that tells SCIDDO how to group samples based on their cellular
characteristics (you can find the below design matrix under `testdata/aux/design_matrix.tsv`):

```
sample       male   female  blood   esc
E014_HUES48   0       1       0      1
E016_HUES64   1       0       0      1
E115_DND41    1       0       1      0
E124_CD14M    0       1       1      0
```

Note that the design matrix has to encode each property in a binary
presence (1) / absence (0) fashion ("one-hot encoding" per property).
To give an example, consider the above case of "male" and "female" samples, which must not be represented
as a single column, say, "sex" with 0 indicating male and 1 indicating female samples. Apart from this
restriction, any number of properties can be included in the design matrix.

Now you can execute the conversion from input to SCIDDO data file as follows:

```
sciddo.py \
    --no-conf-dump \  # do not dump config as json, global option
    --workers 2 \  # use 2 CPU cores, global option
    convert \  # SCIDDO run mode: convert, below follow run mode specific options
    --state-seg testdata/segmentation \  # path to state segmentation files
    --chrom-sizes testdata/aux/hg19.chrom.sizes \  # path to file containing chromosome sizes
    --state-labels testdata/model/brwoserlabelmap_18_core_K27ac.tab \  # state label file
    --state-colors testdata/model/colormap_18_core_K27ac.tab \  # state color file
    --design-matrix testdata/aux/design_matrix.tsv \
    --model-emissions testdata/model/model_18_core_K27ac.txt \  # ChromHMM model file
    --seg-format ChromHMM \  # tells SCIDDO to expect ChromHMM output files (segmentations, emissions etc.)
    --emission-format model \  # the file specified via "model-emissions" is *the* ChromHMM model file
    --chrom-filter "chr10"  # IMPORTANT NOTE - see below
    --output testdata.h5
```

Note regarding chrom filter: since the chromosome size file contains all hg19 human chromosomes, but the state
segmentation files have been reduced to chromosome 10, it is important to filter for chromosome 10 only because
SCIDDO expects all chromosomes in the chromosome size file to also be present (= have chromatin state segments)
in the input data files.
The same effect could be realized by reducing the chromosome size file to just chr10 and omitting the filter option.

### Step 2: compute basic statistics
Strictly speaking, computing statistics is not always required (only for data-derived scoring schemes), but it affords
already a basic summary of the data set and is very quick to complete.

```
sciddo.py \
    --no-conf-dump \
    --workers 2 \
    stats \
    --counts \  # count-based statistics
    --agreement \  # agreement between all samples in the data set
    --sciddo-data testdata.h5
```

Running this on the test data set will issue a warning:

```
WARNING: Identified 0 sample pairs for comparison replicates - is that correct? Skipping to next...
```

Replicates in the SCIDDO data set are identified by selecting identical rows from the design matrix,
which do not exist in this case as there is 1 male and 1 female sample per cell lineage. We hence already
know that we could not use the (anyway experimental) replicate-derived scoring scheme for this data set
(see `--help` for command `score` for details).

### Step 3: add a scoring matrix
Adding a chromatin state scoring matrix is mandatory before scanning for differential
chromatin domains. A scoring matrix can be defined as a simple text file (discussed in Part 2 below),
or inferred from the data or model. We are going to illustrate the latter case
and use the chromatin state model emissions to define a scoring scheme:

```
sciddo.py \
    --no-conf-dump \
    score \
    --add-scoring emission \  # taken from the state emission probabilities read during the initial "convert" step
    --treat-background penalized \  # see help for details
    --sciddo-data testdata.h5
```

This will create a scoring matrix based on the chromatin state emission probabilities and store it under the name
"penem" (for penalized emission) in the data set (you can set the `--debug` option to see the name of data- or
model-derived scoring matrices added to the data set).

### Step 4: scan data for differential chromatin domains
The main use case intended for SCIDDO is the identification of differential chromatin domains between (small)
groups of homogenous samples (i.e., high-quality biological replicates). Since the test data set does not
perfectly fit that use case, the option to merge candidate regions into segments ideally representative
of differential chromatin marking for the entire group of samples, is only used here for illustrative purposes:

```
sciddo.py \
    --no-conf-dump \
    --workers 2 \
    scan \
    --sciddo-data testdata.h5 \
    --run-out test_run_penem.h5 \
    --scoring penem \  # could be omitted since there is only one scoring matrix in the data set
    --adjust-group-length linear [adaptive] \  # set adaptive for high-quality data sets
    --merge-segments \  # set for high-quality data sets
    --compute-raw-stats 0 \  # can be omitted for high-quality data sets
    --compute-merged-stats 0 \  # set for high-quality data sets where sample-level DCDs are merged
    --select-groups \
    --group1 blood \  # could follow the same principle for "male vs female" comparison
    --group2 esc
```

### Step 5: dump results to table or BED format output
A SCIDDO data set and the output of a SCIDDO run are stored as [hierarchical data format files (HDF5)](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
for efficiency reasons. This also simplifies data sharing, but HDF files are not generally compatible with common
bioinformatics command line tools for downstream processing of the differential chromatin domain
candidate regions.

Dumping the contents of SCIDDO data or run output files to simple text files is straightforward. Let's dump the results
of the previous DCD scanning step to a tabular (BED-like) text output file:

```
sciddo.py
    --no-conf-dump \
    dump \
    --data-file test_run_penem.h5 \
    --output test_run_penem_raw-DCDs.tsv \ 
    --data-type raw \  # "raw" candidate regions were not merged between individual sample comparisons into "segments"
```

The above command dumps a table that will be accepted by most tools as BED file input (following the common
"chromosome, start, end, name" layout to describe regions of interest). To limit the output to only the essential fields,
you can add the option `--limit-bed-output`. In either case, the first five columns adhere to the BED standard:
chromosome, start, end, (unique) name, score. Note that raw DCD candidate regions (or merged segments) have no orientation,
hence no strand indicator (`+, -, .` in BED files) is given. The remaining fields in the limited output
list the Expect value (E-value, "lower is better"), the (log) p-value ("higher is better"),
the region score normalized by the length, and the two samples where the respective region was identified.
The name of each region contains also a rank information that indicates how many regions score better.
As example, the name of the region `HSP_chr10_L0_R0709` indicates that `R0709 (/100) = 70.9%` of the other regions have
a higher score; for the region `HSP_chr10_L26_R0050`, only `R0050 (/100) = 5%` of the other regions score better.
The above should results in a BED-like file with 4091 candidate regions for differential chromatin marking:

```
$ wc -l test_run_penem_raw-DCD.tsv
4092  # note that there is a header line
```

## Part 2: examples for user customizations
In the second part of the tutorial, we exemplify a couple of options to customize SCIDDO's behavior. All examples below
assume that the first part of the tutorial has been completed (and, similarly, illustrate only functionality, but do not
claim to showcase interesting biology found in the test data set).

### Example 1: split output by chromatin state dynamics ("chromatin dynamics filter")
Based on Step 5 in Part 1 of the tutorial, the output can be further limited to only include certain chromatin
state transitions between samples (but these transitions are nevertheless contained in DCD candidate regions).
The chromatin dynamics filter is currently only supported for merged segments, and cannot be used on SCIDDO
run output files that only contain "raw" DCD candidate regions.
Let's assume we are only interested in switches from "active enhancers" (states 9 and 10) to
"Polycomb-repressed" states (16 and 17).

```
sciddo.py
    --no-conf-dump \
    dump \
    --data-file testdata.h5 \  # note that we need the data file to know which states are part of a DCD
    --support-file test_run_penem.h5 \
    --output test_run_penem_split-DCD.enh-to-pcr.tsv \ 
    --data-type dynamics \  # select chromatin dynamics filter
    --split-segments \  # output splits instead of complete regions
    --from-states 9 10 \  # active enhancer states
    --to-states 16 17 \  # Polycomb-repressed states
    --add-inverse  # add both directions for state switches
    --limit-bed-output
```

The resulting BED-like table contains potentially overlapping regions since the chromatin dynamics filter
operates on the level of individual sample comparisons. However, by default, it retains only a single region
per group comparison in case sample comparisons between groups resulted in identical regions. To keep the full
information and all duplicates, e.g., if you need to know exactly which two samples contained the respective
chromatin state switch, you can add the option `--keep-duplicates` to the above command.

**Important note**: the statistics in the "split" output file pertain to the full-length DCDs, and are only kept
for reference. The chromatin state transition splits can be as short as a single genomic bin, and are thus not
amenable to the same statistical evaluation as the full-length DCDs.

### Example 2: add a custom scoring matrix to the data set
If a particular study design requires a tailored scoring matrix to score chromatin state differences, it is
easiest to adapt an existing scoring matrix. For training purposes, let's examine if our test data set
already contains a scoring matrix to be used for that purpose (we know it does...):

```
$ sciddo.py \
    --no-conf-dump \
    dump \
    --data-file testdata.h5 \
    --data-type info \  # simply list contents of data file

/metadata/basefreq
/metadata/chromosomes
/metadata/design
/metadata/emissions
/metadata/inputs
/metadata/states
/metadata/transitions/singletons
/scoring/penem/matrix  # here it is, the "penalized emission" scoring matrix created in Part 1 - Step 3
/scoring/penem/parameters
/state/E014_HUES48/chr10
/state/E016_HUES64/chr10
/state/E115_DND41/chr10
/state/E124_CD14M/chr10
/statistic/agreement/raw
/statistic/agreement/score
/statistic/counts/composition/E014_HUES48
/statistic/counts/composition/E016_HUES64
/statistic/counts/composition/E115_DND41
/statistic/counts/composition/E124_CD14M
/statistic/counts/transitions/singletons/chr10
/statistic/counts/transitions/singletons/genome
```

We dump that scoring matrix to a text file...

```
sciddo.py \
    --no-conf-dump \
    dump \
    --data-file testdata.h5 \
    --data-type metadata \  # design glitch: a scoring matrix is internally handled like metadata, but is prefixed with /scoring...
    --output penem.tsv \
    --dump-metadata /scoring/penem/matrix
```

and devise a new scoring scheme as follows:

- default score: -2
- bivalent states 14 and 15 to active states 1, 9, 10: 4 (and vice versa)
- bivalent states 14 and 15 to repressed states 16 and 17: 4 (and vice versa)
- background state 18 to any other state: -6

The new scoring matrix needs to be saved as a tab-separated text file (say, "custom.tsv") and can be added to the SCIDDO data set as follows:

```
sciddo.py \
    --no-conf-dump \
    score \
    --scoring-file custom.tsv \
    --sciddo-data testdata.h5
```

Next, the same scan for DCDs as in Part 1 - Step 4 can be performed, but with the new scoring matrix `--scoring custom`. After a successful run
and dumping the results to a text table, we find 429 DCDs "focused" on bivalent domains that change to an active or repressed state:

```
$ wc -l test_run_custom_DCD.tsv
430  # note that there is a header line
```
