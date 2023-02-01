/*
============================================================================
Filename    : pi.c
Author      : Your names goes here
SCIPER		: Your SCIPER numbers
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include <omp.h>

double calculate_pi (int num_threads, int samples);

int main (int argc, const char *argv[]) {

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
		printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
	}

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
    double pi;
    int sum = 0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        rand_gen rg = init_rand();
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < samples; i++) {
            double x = next_rand(rg);
            double y = next_rand(rg);

            if (x * x + y * y <= 1) {
                sum += 1;
            }
        }
        free_rand(rg);
    }

    pi = sum * 4.0 / samples;
    return pi;
}
