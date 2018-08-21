#include<stdio.h>
#include<stdlib.h>
#include "karlin.h"

main()
{
    printf("Running Karlin subroutine with example data\n");
	int lowest = -2;
	int highest = 3;
	int exc = 1;

	double score_probs[6] = { 0.7, 0.0, 0.1, 0.0, 0.0, 0.2 };
    double* ptr_probs = score_probs;

    double lambda[1] = {0.};
    double* ptr_lambda = lambda;

    double entropy[1] = {0.};
    double* ptr_entropy = entropy;

    double kappa[1] = {0.};
    double* ptr_kappa = kappa;

    /*
    int8_t ComputeKarlinAltschulParameters(double* scorefreq, int16_t obs_min, int16_t obs_max,
                                        int16_t score_min, int16_t score_max,
                                        double* lambda, double* entropy, double* k);

    */
    printf("Calling...\n");
    exc = ComputeKarlinAltschulParameters(ptr_probs, lowest, highest, -8, 8, ptr_lambda, ptr_entropy, ptr_kappa);

    printf("Return code: %u\n", exc);
    printf("Lambda estimate: %.5f\n", lambda[0]);  /* ~0.33 */
    printf("Entropy estimate: %.5f\n", entropy[0]); /* ~0.3 */
    printf("Kappa estimate: %.5f\n", kappa[0]); /* ~0.154 */

	exit(exc);
}
