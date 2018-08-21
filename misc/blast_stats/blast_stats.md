# Implementation notes on sequence alignment statistics as used in BLAST

All code excerpts were taken from BLAST release v2.7.1, which can be found
on the [NCBI BLAST FTP server](ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/)

The BLAST source code has been released into the public domain; author of the relevant
source code files is Tom Madden. All code excerpts below were extracted from the
files `blast_stat.h` and `blast_stat.c`.

The `karlin.c` file is a minimal, stand-alone version of the routines to compute
lambda, H and K. This C source file is compiled into a shared library object that
is used in the SCIDDO Python code.

The file `test_karlin.c` can be compiled to run a test of the stand-alone version
with expected results (taken from the comment in the original code; see, e.g., PDF
of developer notes below, section A.1).

## Compilation instructions

To compile `karlin.c` as shared libray object:

```bash
gcc -fPIC -shared -o karlin.so karlin.c
```

To compile `test_karlin.c` as executable:

```bash
gcc karlin.c test_karlin.c -o test_karlin -lm
```

Compilation was tested on a Debian 7.11 system with gcc 4.7.2

## Developer notes

There is a PDF document with developer notes that can be found on the
[NCBI BLAST FTP server](ftp://ftp.ncbi.nlm.nih.gov/blast/documents/developer/scoring.pdf).
This document is included in this repository for reference (renamed: Gertz_2005_BLAST_scoring.pdf).

## blast_stat.h

Struct that defines a BLAST scoring matrix:

```C
/** Scoring matrix used in BLAST */
typedef struct SBlastScoreMatrix {
    int** data;         /**< actual scoring matrix data, stored in row-major
                          form */
    size_t ncols;       /**< number of columns */
    size_t nrows;       /**< number of rows */
    double* freqs;      /**< array of assumed matrix background frequencies -RMH-*/
    double lambda;      /**< derived value of the matrix lambda -RMH- */
} SBlastScoreMatrix;
```

Struct that defines all score attributes:

```C
/** Holds score frequencies used in calculation
of Karlin-Altschul parameters for an ungapped search.
*/
typedef struct Blast_ScoreFreq {
    Int4         score_min; /**< lowest allowed scores */
    Int4         score_max; /**< highest allowed scores */
    Int4         obs_min;   /**< lowest observed (actual) scores */
    Int4         obs_max;   /**< highest observed (actual) scores */
    double       score_avg; /**< average score, must be negative for local alignment. */
    double*      sprob0;    /**< arrays for frequency of given score */
    double*      sprob;     /**< arrays for frequency of given score, shifted down by score_min. */
} Blast_ScoreFreq;
```

Struct that defines letter requencies:

```C
/**
Stores the letter frequency of a sequence or database.
*/
typedef struct Blast_ResFreq {
    Uint1   alphabet_code;    /**< indicates alphabet. */
    double* prob;       /**< letter probs, (possible) non-zero offset. */
    double* prob0;            /**< probs, zero offset. */
} Blast_ResFreq;
```

Struct that defines (all possible?) scores plus additional information used in BLAST:

```C
/** Structure used for scoring calculations.
*/
typedef struct BlastScoreBlk {
   Boolean     protein_alphabet; /**< TRUE if alphabet_code is for a
protein alphabet (e.g., ncbistdaa etc.), FALSE for nt. alphabets. */
   Uint1    alphabet_code; /**< NCBI alphabet code. */
   Int2     alphabet_size;  /**< size of alphabet. */
   Int2     alphabet_start;  /**< numerical value of 1st letter. */
   char*    name;           /**< name of scoring matrix. */
   ListNode*   comments;    /**< Comments about scoring matrix. */
   SBlastScoreMatrix* matrix;   /**< scoring matrix data */
   SPsiBlastScoreMatrix* psi_matrix;    /**< PSSM and associated data. If this
                                         is not NULL, then the BLAST search is
                                         position specific (i.e.: PSI-BLAST) */
   Boolean  matrix_only_scoring;  /**< Score ungapped/gapped alignment only
                                       using the matrix parameters and
                                       with raw scores. Ignore
                                       penalty/reward and do not report
                                       Karlin-Altschul stats.  This is used
                                       by the rmblastn program. -RMH- */
   Boolean complexity_adjusted_scoring; /**< Use cross_match-like complexity
                                           adjustment on raw scores. -RMH- */
   Int4  loscore;   /**< Min.  substitution scores */
   Int4  hiscore;   /**< Max. substitution scores */
   Int4  penalty;   /**< penalty for mismatch in blastn. */
   Int4  reward;    /**< reward for match in blastn. */
        double  scale_factor; /**< multiplier for all cutoff and dropoff scores */
   Boolean     read_in_matrix; /**< If TRUE, matrix is read in, otherwise
               produce one from penalty and reward above. @todo should this be
                an allowed way of specifying the matrix to use? */
   Blast_ScoreFreq** sfp;  /**< score frequencies for scoring matrix. */
   /* kbp & kbp_gap are ptrs that should be set to kbp_std, kbp_psi, etc. */
   Blast_KarlinBlk** kbp;  /**< Karlin-Altschul parameters. Actually just a placeholder. */
   Blast_KarlinBlk** kbp_gap; /**< K-A parameters for gapped alignments.  Actually just a placeholder. */
   Blast_GumbelBlk* gbp;  /**< Gumbel parameters for FSC. */
   /* Below are the Karlin-Altschul parameters for non-position based ('std')
   and position based ('psi') searches. */
   Blast_KarlinBlk **kbp_std,  /**< K-A parameters for ungapped alignments */
                    **kbp_psi,       /**< K-A parameters for position-based alignments. */
                    **kbp_gap_std,  /**< K-A parameters for std (not position-based) alignments */
                    **kbp_gap_psi;  /**< K-A parameters for psi alignments. */
   Blast_KarlinBlk*  kbp_ideal;  /**< Ideal values (for query with average database composition). */
   Int4 number_of_contexts;   /**< Used by sfp and kbp, how large are these*/
   Uint1*   ambiguous_res; /**< Array of ambiguous res. (e.g, 'X', 'N')*/
   Int2     ambig_size, /**< size of array above. FIXME: not needed here? */
         ambig_occupy;  /**< How many occupied? */
   Boolean  round_down; /**< Score must be rounded down to nearest even score if odd. */
} BlastScoreBlk;
```

## blast_stat.c

### Calculating score frequencies

Score frequencies are required as input to the parameter
estimation routines for lambda, K etc.

Start - init and sanity checks

```C
/** Calculates the score frequencies.
 *
 * @param sbp object with scoring information [in]
 * @param sfp object to hold frequency information [in|out]
 * @param rfp1 letter frequencies for first sequence (query) [in]
 * @param rfp2 letter frequencies for second sequence (database) [in]
 * @return zero on success
 */
static Int2
BlastScoreFreqCalc(const BlastScoreBlk* sbp, Blast_ScoreFreq* sfp, Blast_ResFreq* rfp1, Blast_ResFreq* rfp2)
{
   Int4 **  matrix;
   Int4  score, obs_min, obs_max;
   double      score_sum, score_avg;
   Int2     alphabet_start, alphabet_end, index1, index2;

   if (sbp == NULL || sfp == NULL)
      return 1;

   if (sbp->loscore < sfp->score_min || sbp->hiscore > sfp->score_max)
      return 1;

   for (score = sfp->score_min; score <= sfp->score_max; score++)
      sfp->sprob[score] = 0.0;
```

Set matrix to score matrix. Iteration over score matrix yields
score for letter pairing "index1 - index2". Then get letter
(background) frequencies (that should be p_i, p_j in the literature)
from Blast_ResFreq objects.

Note: necessity for check `score >= sbp->loscore` is unclear

```C
   matrix = sbp->matrix->data;

   alphabet_start = sbp->alphabet_start;
   alphabet_end = alphabet_start + sbp->alphabet_size;
   for (index1=alphabet_start; index1<alphabet_end; index1++)
   {
      for (index2=alphabet_start; index2<alphabet_end; index2++)
      {
         score = matrix[index1][index2];
         if (score >= sbp->loscore)
         {
            sfp->sprob[score] += rfp1->prob[index1] * rfp2->prob[index2];
         }
      }
   }
```

Collected all raw score frequencies. By the way the scores
were collected, it is possible that the score frequencies
do not sum to 1. `score_sum` is a sum over all score
probabilities. Score probabilities are then normalized
and the average score is calculated.

```C
   score_sum = 0.;
   obs_min = obs_max = BLAST_SCORE_MIN;
   for (score = sfp->score_min; score <= sfp->score_max; score++)
   {
      if (sfp->sprob[score] > 0.)
      {
         score_sum += sfp->sprob[score];
         obs_max = score;
         if (obs_min == BLAST_SCORE_MIN)
            obs_min = score;
      }
   }
   sfp->obs_min = obs_min;
   sfp->obs_max = obs_max;

   score_avg = 0.0;
   if (score_sum > 0.0001 || score_sum < -0.0001)
   {
      for (score = obs_min; score <= obs_max; score++)
      {
         sfp->sprob[score] /= score_sum;
         score_avg += score * sfp->sprob[score];
      }
   }
   sfp->score_avg = score_avg;

   return 0;
}
```
