/*
Code extracted from NCBI BLAST v2.7.1 (in May 2018)
by
Peter Ebert
*/


/* $Id: blast_stat.c 505944 2016-06-30 12:29:20Z madden $
 * ===========================================================================
 *
 *                            PUBLIC DOMAIN NOTICE
 *               National Center for Biotechnology Information
 *
 *  This software/database is a "United States Government Work" under the
 *  terms of the United States Copyright Act.  It was written as part of
 *  the author's official duties as a United States Government employee and
 *  thus cannot be copyrighted.  This software/database is freely available
 *  to the public for use. The National Library of Medicine and the U.S.
 *  Government have not placed any restriction on its use or reproduction.
 *
 *  Although all reasonable efforts have been taken to ensure the accuracy
 *  and reliability of the software and data, the NLM and the U.S.
 *  Government do not and cannot warrant the performance or results that
 *  may be obtained by using this software or data. The NLM and the U.S.
 *  Government disclaim all warranties, express or implied, including
 *  warranties of performance, merchantability or fitness for any particular
 *  purpose.
 *
 *  Please cite the author in any work or product based on this material.
 *
 * ===========================================================================
 *
 * Author: Tom Madden
 *
 */

/** @file blast_stat.c
 * Functions to calculate BLAST probabilities etc.
 * Detailed Contents:
 *
 * - allocate and deallocate structures used by BLAST to calculate
 * probabilities etc.
 *
 * - calculate residue frequencies for query and "average" database.
 *
 * - read in matrix or load it from memory.
 *
 *  - calculate sum-p from a collection of HSP's, for both the case
 *   of a "small" gap and a "large" gap, when give a total score and the
 *   number of HSP's.
 *
 * - calculate expect values for p-values.
 *
 * - calculate pseuod-scores from p-values.
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

/* PE_COMMENT
To make this standalone, needed to
remove the dependency on the other
BLAST library parts

#include <algo/blast/core/blast_stat.h>
#include <algo/blast/core/ncbi_math.h>
#include "boost_erf.h"
#include "blast_psi_priv.h"

The "entry point" function at the end of
this file has been added to get all
parameter estimates with a single call
*/

/* PE COMMENT
Following definition copied to make
this run
*/

#ifndef ABS
/** returns absolute value of a (|a|) */
#define ABS(a)	((a)>=0?(a):-(a))
#endif

/* PE_COMMENT
The min/max score is set to comparatively
 low values, which should be good enough
 for the intended use case
*/
#define BLAST_SCORE_MIN INT16_MIN   /**< minimum allowed score (for one letter comparison). */
#define BLAST_SCORE_MAX INT16_MAX   /**< maximum allowed score (for one letter comparison). */

#define BLAST_SCORE_RANGE_MAX   (BLAST_SCORE_MAX - BLAST_SCORE_MIN) /**< maximum allowed range of BLAST scores. */

/****************************************************************************
 * For more accuracy in the calculation of K, set K_SUMLIMIT to 0.00001.
 * For high speed in the calculation of K, use a K_SUMLIMIT of 0.001
 * Note:  statistical significance is often not greatly affected by the value
 * of K, so high accuracy is generally unwarranted.
 *****************************************************************************/
#define BLAST_KARLIN_K_SUMLIMIT_DEFAULT 0.0001 /**< K_SUMLIMIT_DEFAULT == sumlimit used in BlastKarlinLHtoK() */

#define BLAST_KARLIN_LAMBDA_ACCURACY_DEFAULT    (1.e-5) /**< LAMBDA_ACCURACY_DEFAULT == accuracy to which Lambda should be calc'd */

#define BLAST_KARLIN_LAMBDA_ITER_DEFAULT        17 /**< LAMBDA_ITER_DEFAULT == no. of iterations in LambdaBis = ln(accuracy)/ln(2)*/

#define BLAST_KARLIN_LAMBDA0_DEFAULT    0.5 /**< Initial guess for the value of Lambda in BlastKarlinLambdaNR */

#define BLAST_KARLIN_K_ITER_MAX 100 /**< upper limit on iterations for BlastKarlinLHtoK */

/** Holds score frequencies used in calculation
 o f Karlin-Altschul* parameters for an ungapped search.
 */
typedef struct Blast_ScoreFreq {
	int16_t         score_min; /**< lowest allowed scores */
	int16_t         score_max; /**< highest allowed scores */
	int16_t         obs_min;   /**< lowest observed (actual) scores */
	int16_t         obs_max;   /**< highest observed (actual) scores */
	double       score_avg; /**< average score, must be negative for local alignment. */
	double*      sprob0;    /**< arrays for frequency of given score */
	double*      sprob;     /**< arrays for frequency of given score, shifted down by score_min. */
} Blast_ScoreFreq;

/** Check that the lo and hi score are within the allowed ranges
 * @param lo the lowest permitted value [in]
 * @param hi the highest permitted value [in]
 * @return zero on success, 1 otherwise
 */

static int8_t
BlastScoreChk(int16_t lo, int16_t hi)
{
	if (lo >= 0 || hi <= 0 ||
		lo < BLAST_SCORE_MIN || hi > BLAST_SCORE_MAX)
		return 1;

	if (hi - lo > BLAST_SCORE_RANGE_MAX)
		return 1;

	return 0;
}

Blast_ScoreFreq*
Blast_ScoreFreqFree(Blast_ScoreFreq* sfp)
{
   if (sfp == NULL)
      return NULL;

   if (sfp->sprob0 != NULL)
      free(sfp->sprob0);
      sfp->sprob0 = NULL;
   free(sfp);
   sfp = NULL;
   return sfp;
}

Blast_ScoreFreq*
Blast_ScoreFreqNew(int16_t score_min, int16_t score_max)
{
   Blast_ScoreFreq*  sfp;
   int16_t  range;

   if (BlastScoreChk(score_min, score_max) != 0)
      return NULL;

   sfp = (Blast_ScoreFreq*) calloc(1, sizeof(Blast_ScoreFreq));
   if (sfp == NULL)
      return NULL;

   range = score_max - score_min + 1;
   sfp->sprob = (double*) calloc(range, sizeof(double));
   if (sfp->sprob == NULL)
   {
      Blast_ScoreFreqFree(sfp);
      return NULL;
   }

   sfp->sprob0 = sfp->sprob;
   sfp->sprob -= score_min;        /* center around 0 */
   sfp->score_min = score_min;
   sfp->score_max = score_max;
   sfp->obs_min = sfp->obs_max = 0;
   sfp->score_avg = 0.0;
   return sfp;
}

double BLAST_Powi(double x, int32_t n)
{
   double   y;

   if (n == 0)
      return 1.;

   if (x == 0.) {
      if (n < 0) {
         return HUGE_VAL;
      }
      return 0.;
   }

   if (n < 0) {
      x = 1./x;
      n = -n;
   }

   y = 1.;
   while (n > 0) {
      if (n & 1)
         y *= x;
      n /= 2;
      x *= x;
   }
   return y;
}

int16_t BLAST_Gcd(int16_t a, int16_t b)
{
	int16_t   c;
	
	b = ABS(b);
	if (b > a)
		c=a, a=b, b=c;
	
	while (b != 0) {
		c = a%b;
		a = b;
		b = c;
	}
	return a;
}

double BLAST_Expm1(double	x)
{
	double	absx = ABS(x);
	
	if (absx > .33)
		return exp(x) - 1.;
	
	if (absx < 1.e-16)
		return x;
	
	return x * (1. + x *
	(1./2. + x * 
	(1./6. + x *
	(1./24. + x * 
	(1./120. + x *
	(1./720. + x * 
	(1./5040. + x *
	(1./40320. + x * 
	(1./362880. + x *
	(1./3628800. + x * 
	(1./39916800. + x *
	(1./479001600. + 
	x/6227020800.))))))))))));
}

/** The following procedure computes K. The input includes Lambda, H,
 *  and an array of probabilities for each score.
 *  There are distinct closed form for three cases:
 *  1. high score is 1 low score is -1
 *  2. high score is 1 low score is not -1
 *  3. low score is -1, high score is not 1
 *
 * Otherwise, in most cases the value is computed as:
 * -exp(-2.0*outerSum) / ((H/lambda)*(exp(-lambda) - 1)
 * The last term (exp(-lambda) - 1) can be computed in two different
 * ways depending on whether lambda is small or not.
 * outerSum is a sum of the terms
 * innerSum/j, where j is denoted by iterCounter in the code.
 * The sum is truncated when the new term innersum/j i sufficiently small.
 * innerSum is a weighted sum of the probabilities of
 * of achieving a total score i in a gapless alignment,
 * which we denote by P(i,j).
 * of exactly j characters. innerSum(j) has two parts
 * Sum over i < 0  P(i,j)exp(-i * lambda) +
 * Sum over i >=0  P(i,j)
 * The terms P(i,j) are computed by dynamic programming.
 * An earlier version was flawed in that ignored the special case 1
 * and tried to replace the tail of the computation of outerSum
 * by a geometric series, but the base of the geometric series
 * was not accurately estimated in some cases.
 *
 * @param sfp object holding scoring frequency information [in]
 * @param lambda a Karlin-Altschul parameter [in]
 * @param H a Karlin-Altschul parameter [in]
 * @return K, another Karlin-Altschul parameter
 */

static double
BlastKarlinLHtoK(Blast_ScoreFreq* sfp, double lambda, double H)
{
    /*The next array stores the probabilities of getting each possible
      score in an alignment of fixed length; the array is shifted
      during part of the computation, so that
      entry 0 is for score 0.  */
    double         *alignmentScoreProbabilities = NULL;
    int16_t            low;    /* Lowest score (must be negative) */
    int16_t            high;   /* Highest score (must be positive) */
    int16_t            range;  /* range of scores, computed as high - low*/
    double          K;      /* local copy of K  to return*/
    int             i;   /*loop index*/
    int             iterCounter; /*counter on iterations*/
    int16_t            divisor; /*candidate divisor of all scores with
                               non-zero probabilities*/
    /*highest and lowest possible alignment scores for current length*/
    int16_t            lowAlignmentScore, highAlignmentScore;
    int16_t            first, last; /*loop indices for dynamic program*/
    register double innerSum;
    double          oldsum, oldsum2;  /* values of innerSum on previous
                                         iterations*/
    double          outerSum;        /* holds sum over j of (innerSum
                                        for iteration j/j)*/

    double          score_avg; /*average score*/
    /*first term to use in the closed form for the case where
      high == 1 or low == -1, but not both*/
    double          firstTermClosedForm;  /*usually store H/lambda*/
    int             iterlimit; /*upper limit on iterations*/
    double          sumlimit; /*lower limit on contributions
                                to sum over scores*/

    /*array of score probabilities reindexed so that low is at index 0*/
    double         *probArrayStartLow;

    /*pointers used in dynamic program*/
    double         *ptrP, *ptr1, *ptr2, *ptr1e;
    double          expMinusLambda; /*e^^(-Lambda) */

    if (lambda <= 0. || H <= 0.) {
        /* Theory dictates that H and lambda must be positive, so
         * return -1 to indicate an error */
        return -1.;
    }

    /*Karlin-Altschul theory works only if the expected score
      is negative*/
    if (sfp->score_avg >= 0.0) {
        return -1.;
    }

    low   = sfp->obs_min;
    high  = sfp->obs_max;
    range = high - low;

    probArrayStartLow = &sfp->sprob[low];
    /* Look for the greatest common divisor ("delta" in Appendix of PNAS 87 of
       Karlin&Altschul (1990) */
    for (i = 1, divisor = -low; i <= range && divisor > 1; ++i) {
        if (probArrayStartLow[i] != 0.0)
            divisor = BLAST_Gcd(divisor, i);
    }

    high   /= divisor;
    low    /= divisor;
    lambda *= divisor;

    range = high - low;

    firstTermClosedForm = H/lambda;
    expMinusLambda      = exp((double) -lambda);

    if (low == -1 && high == 1) {
        K = (sfp->sprob[low*divisor] - sfp->sprob[high*divisor]) *
            (sfp->sprob[low*divisor] - sfp->sprob[high*divisor]) / sfp->sprob[low*divisor];
        return(K);
    }

    if (low == -1 || high == 1) {
        if (high != 1) {
            score_avg = sfp->score_avg / divisor;
            firstTermClosedForm
                = (score_avg * score_avg) / firstTermClosedForm;
        }
        return firstTermClosedForm * (1.0 - expMinusLambda);
    }

    sumlimit  = BLAST_KARLIN_K_SUMLIMIT_DEFAULT;
    iterlimit = BLAST_KARLIN_K_ITER_MAX;

    alignmentScoreProbabilities =
        (double *)calloc((iterlimit*range + 1), sizeof(*alignmentScoreProbabilities));
    if (alignmentScoreProbabilities == NULL)
        return -1.;

    outerSum = 0.;
    lowAlignmentScore = highAlignmentScore = 0;
    alignmentScoreProbabilities[0] = innerSum = oldsum = oldsum2 = 1.;

    for (iterCounter = 0;
         ((iterCounter < iterlimit) && (innerSum > sumlimit));
         outerSum += innerSum /= ++iterCounter) {
        first = last = range;
        lowAlignmentScore  += low;
        highAlignmentScore += high;
        /*dynamic program to compute P(i,j)*/
        for (ptrP = alignmentScoreProbabilities +
                 (highAlignmentScore-lowAlignmentScore);
             ptrP >= alignmentScoreProbabilities;
             *ptrP-- =innerSum) {
            ptr1  = ptrP - first;
            ptr1e = ptrP - last;
            ptr2  = probArrayStartLow + first;
            for (innerSum = 0.; ptr1 >= ptr1e; ) {
                innerSum += *ptr1  *  *ptr2;
		ptr1--;
		ptr2++;
            }
            if (first)
                --first;
            if (ptrP - alignmentScoreProbabilities <= range)
                --last;
        }
        /* Horner's rule */
        innerSum = *++ptrP;
        for( i = lowAlignmentScore + 1; i < 0; i++ ) {
            innerSum = *++ptrP + innerSum * expMinusLambda;
        }
        innerSum *= expMinusLambda;

        for (; i <= highAlignmentScore; ++i)
            innerSum += *++ptrP;
        oldsum2 = oldsum;
        oldsum  = innerSum;
    }

#ifdef ADD_GEOMETRIC_TERMS_TO_K
    /*old code assumed that the later terms in sum were
      asymptotically comparable to those of a geometric
      progression, and tried to speed up convergence by
      guessing the estimated ratio between sucessive terms
      and using the explicit terms of a geometric progression
      to speed up convergence. However, the assumption does not
      always hold, and convergenece of the above code is fast
      enough in practice*/
    /* Terms of geometric progression added for correction */
    {
        double     ratio;  /* fraction used to generate the
                                   geometric progression */

        ratio = oldsum / oldsum2;
        if (ratio >= (1.0 - sumlimit*0.001)) {
            K = -1.;
            if (alignmentScoreProbabilities != NULL)
                free(alignmentScoreProbabilities);
                alignmentScoreProbabilities = NULL;
            return K;
        }
        sumlimit *= 0.01;
        while (innerSum > sumlimit) {
            oldsum   *= ratio;
            outerSum += innerSum = oldsum / ++iterCounter;
        }
    }
#endif

    K = -exp((double)-2.0*outerSum) /
             (firstTermClosedForm*BLAST_Expm1(-(double)lambda));

    if (alignmentScoreProbabilities != NULL)
        free(alignmentScoreProbabilities);
        alignmentScoreProbabilities = NULL;

    return K;
}


/**
 * Find positive solution to
 *
 *     sum_{i=low}^{high} exp(i lambda) * probs[i] = 1.
 *
 * Note that this solution does not exist unless the average score is
 * negative and the largest score that occurs with nonzero probability
 * is positive.
 *
 * @param probs         probabilities of a score occurring
 * @param d             the gcd of the possible scores. This equals 1 if
 *                      the scores are not a lattice
 * @param low           the lowest possible score that occurs with
 *                      nonzero probability
 * @param high          the highest possible score that occurs with
 *                      nonzero probability.
 * @param lambda0       an initial guess for lambda
 * @param tolx          the tolerance to which lambda must be computed
 * @param itmax         the maximum number of times the function may be
 *                      evaluated
 * @param maxNewton     the maximum permissible number of Newton
 *                      iterations; after that the computation will proceed
 *                      by bisection.
 * @param *itn          the number of iterations needed to compute Lambda,
 *                      or itmax if Lambda could not be computed.
 *
 * Let phi(lambda) =  sum_{i=low}^{high} exp(i lambda) - 1. Then
 * phi(lambda) may be written
 *
 *     phi(lamdba) = exp(u lambda) f( exp(-lambda) )
 *
 * where f(x) is a polynomial that has exactly two zeros, one at x = 1
 * and one at x = exp(-lamdba).  It is simpler to solve this problem
 * in x = exp(-lambda) than it is to solve it in lambda, because we
 * know that for x, a solution lies in [0,1], and because Newton's
 * method is generally more stable and efficient for polynomials than
 * it is for exponentials.
 *
 * For the most part, this function is a standard safeguarded Newton
 * iteration: define an interval of uncertainty [a,b] with f(a) > 0
 * and f(b) < 0 (except for the initial value b = 1, where f(b) = 0);
 * evaluate the function and use the sign of that value to shrink the
 * interval of uncertainty; compute a Newton step; and if the Newton
 * step suggests a point outside the interval of uncertainty or fails
 * to decrease the function sufficiently, then bisect.  There are
 * three further details needed to understand the algorithm:
 *
 * 1)  If y the unique solution in [0,1], then f is positive to the left of
 *     y, and negative to the right.  Therefore, we may determine whether
 *     the Newton step -f(x)/f'(x) is moving toward, or away from, y by
 *     examining the sign of f'(x).  If f'(x) >= 0, we bisect instead
 *     of taking the Newton step.
 * 2)  There is a neighborhood around x = 1 for which f'(x) >= 0, so
 *     (1) prevents convergence to x = 1 (and for a similar reason
 *     prevents convergence to x = 0, if the function is incorrectly
 *     called with probs[high] == 0).
 * 3)  Conditions like  fabs(p) < lambda_tolerance * x * (1-x) are used in
 *     convergence criteria because these values translate to a bound
 *     on the relative error in lambda.  This is proved in the
 *     "Blast Scoring Parameters" document that accompanies the BLAST
 *     code.
 *
 * The iteration on f(x) is robust and doesn't overflow; defining a
 * robust safeguarded Newton iteration on phi(lambda) that cannot
 * converge to lambda = 0 and that is protected against overflow is
 * more difficult.  So (despite the length of this comment) the Newton
 * iteration on f(x) is the simpler solution.
 */
static double
NlmKarlinLambdaNR(double* probs, int16_t d, int16_t low, int16_t high, double lambda0,
                  double tolx, int16_t itmax, int16_t maxNewton, int16_t * itn )
{
  int16_t k;
  double x0, x, a = 0, b = 1;
  double f = 4;  /* Larger than any possible value of the poly in [0,1] */
  int16_t isNewton = 0; /* we haven't yet taken a Newton step. */

  assert( d > 0 );

   x0 = exp( -lambda0 );
  x = ( 0 < x0 && x0 < 1 ) ? x0 : .5;

  for( k = 0; k < itmax; k++ ) { /* all iteration indices k */
    int16_t i;
    double g, fold = f;
    int16_t wasNewton = isNewton; /* If true, then the previous step was a */
                              /* Newton step */
    isNewton  = 0;            /* Assume that this step is not */

    /* Horner's rule for evaluating a polynomial and its derivative */
    g = 0;
    f = probs[low];
    for( i = low + d; i < 0; i += d ) {
      g = x * g + f;
      f = f * x + probs[i];
    }
    g = x * g + f;
    f = f * x + probs[0] - 1;
    for( i = d; i <= high; i += d ) {
      g = x * g + f;
      f = f * x + probs[i];
    }
    /* End Horner's rule */

    if( f > 0 ) {
      a = x; /* move the left endpoint */
    } else if( f < 0 ) {
      b = x; /* move the right endpoint */
    } else { /* f == 0 */
      break; /* x is an exact solution */
    }
    if( b - a < 2 * a * ( 1 - b ) * tolx ) {
      /* The midpoint of the interval converged */
      x = (a + b) / 2; break;
    }

    if( k >= maxNewton ||
        /* If convergence of Newton's method appears to be failing; or */
            ( wasNewton && fabs( f ) > .9 * fabs(fold) ) ||
        /* if the previous iteration was a Newton step but didn't decrease
         * f sufficiently; or */
        g >= 0
        /* if a Newton step will move us away from the desired solution */
        ) { /* then */
      /* bisect */
      x = (a + b)/2;
    } else {
      /* try a Newton step */
      double p = - f/g;
      double y = x + p;
      if( y <= a || y >= b ) { /* The proposed iterate is not in (a,b) */
        x = (a + b)/2;
      } else { /* The proposed iterate is in (a,b). Accept it. */
        isNewton = 1;
        x = y;
        if( fabs( p ) < tolx * x * (1-x) ) break; /* Converged */
      } /* else the proposed iterate is in (a,b) */
    } /* else try a Newton step. */
  } /* end for all iteration indices k */
   *itn = k;
  return -log(x)/d;
}


double
Blast_KarlinLambdaNR(Blast_ScoreFreq* sfp, double initialLambdaGuess)
{
   int16_t  low;        /* Lowest score (must be negative)  */
   int16_t  high;       /* Highest score (must be positive) */
   int16_t     itn;
   int16_t  i, d;
   double*  sprob;
   double   returnValue;

   low = sfp->obs_min;
   high = sfp->obs_max;
   if (sfp->score_avg >= 0.) {   /* Expected score must be negative */
      return -1.0;
   }
   if (BlastScoreChk(low, high) != 0) return -1.;

   sprob = sfp->sprob;
   /* Find greatest common divisor of all scores */
   for (i = 1, d = -low; i <= high-low && d > 1; ++i) {
      if (sprob[i+low] != 0.0) {
         d = BLAST_Gcd(d, i);
      }
   }
   returnValue =
      NlmKarlinLambdaNR( sprob, d, low, high,
                           initialLambdaGuess,
                           BLAST_KARLIN_LAMBDA_ACCURACY_DEFAULT,
                     20, 20 + BLAST_KARLIN_LAMBDA_ITER_DEFAULT, &itn );


   return returnValue;
}



/** Calculate H, the relative entropy of the p's and q's
 *
 * @param sfp object containing scoring frequency information [in]
 * @param lambda a Karlin-Altschul parameter [in]
 * @return H, a Karlin-Altschul parameter
 */
static double
BlastKarlinLtoH(Blast_ScoreFreq* sfp, double lambda)
{
   int16_t  score;
   double   H, etonlam, sum, scale;

   double *probs = sfp->sprob;
   int16_t low   = sfp->obs_min,  high  = sfp->obs_max;

   if (lambda < 0.) {
      return -1.;
   }
   if (BlastScoreChk(low, high) != 0) return -1.;

   etonlam = exp( - lambda );
  sum = low * probs[low];
  for( score = low + 1; score <= high; score++ ) {
    sum = score * probs[score] + etonlam * sum;
  }

  scale = BLAST_Powi( etonlam, high );
  if( scale > 0.0 ) {
    H = lambda * sum/scale;
  } else { /* Underflow of exp( -lambda * high ) */
    H = lambda * exp( lambda * high + log(sum) );
  }
   return H;
}

/*
* =================================================
* THE FOLLOWING FUNCTION IS _NOT_ TAKEN FROM THE
* BLAST SOURCE CODE - JUST A CUSTOM ENTRY POINT
* =================================================
*/

int8_t
ComputeKarlinAltschulParameters(
    double* scorefreq, /* Score frequencies, must sum to 1 */
    int16_t obs_min, /* minimum observed score */
    int16_t obs_max, /* maximum observed score */
    int16_t score_min, /* minimum boundary on scores (not minimum observed) */
    int16_t score_max, /* maximum boundary on scores (not maximum observed) */
    double* lambda, /* pointer to parameter lambda */
    double* entropy, /* pointer to parameter H, entropy */
    double* k /* pointer to parameter K */)
{
    double score_avg = 0.;
    double calc_tmp = 0.;
    int16_t iter;
    Blast_ScoreFreq* sfp;

    for (iter = obs_min; iter <= obs_max; iter++)
    {
        calc_tmp = iter;
        score_avg += calc_tmp * scorefreq[iter - obs_min];
    }

    sfp = Blast_ScoreFreqNew(score_min, score_max);
    sfp->obs_min = obs_min;
    sfp->obs_max = obs_max;
    sfp->score_avg = score_avg;

    /*
    initialize array to all zeros
     this is analogous to what is done
     in BlastScoreFreqCalc (not included here,
     see BLAST source)
    */
    for (iter = sfp->score_min; iter <= sfp->score_max; iter++)
    {
        sfp->sprob[iter] = 0.0;
    }

    /*
    set relevant entries to non-zero
    */
    for (iter = obs_min; iter <= obs_max; iter++)
    {
        sfp->sprob[iter] = scorefreq[iter - obs_min];
    }

    *lambda = Blast_KarlinLambdaNR(sfp, BLAST_KARLIN_LAMBDA0_DEFAULT);
    *entropy = BlastKarlinLtoH(sfp, *lambda);
    *k = BlastKarlinLHtoK(sfp, *lambda, *entropy);

    sfp = Blast_ScoreFreqFree(sfp);

    return 0;
}
