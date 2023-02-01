/*
============================================================================
Filename    : pi.c
Author      : Your names goes here
SCIPER		: Your SCIPER numbers
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utility.h"

int perform_buckets_computation(int, int, int);

typedef struct {
    int c;
    int pad[15];
} PaddingHist;

int main (int argc, const char *argv[]) {
    int num_threads, num_samples, num_buckets;

    if (argc != 4) {
		printf("Invalid input! Usage: ./sharing <num_threads> <num_samples> <num_buckets> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        num_buckets = atoi(argv[3]);
	}
    
    set_clock();
    perform_buckets_computation(num_threads, num_samples, num_buckets);

    printf("Using %d threads: %d operations completed in %.4gs.\n", num_threads, num_samples, elapsed_time());
    return 0;
}

int perform_buckets_computation(int num_threads, int num_samples, int num_buckets) {
    volatile int *histogram = (int*) calloc(num_buckets, sizeof(int));
    volatile PaddingHist tmp_hist[num_threads][num_buckets];
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        rand_gen rg = init_rand();
        #pragma omp for 
        for(int i = 0; i < num_samples; i++){
            int val = next_rand(rg) * num_buckets;
            int tid = omp_get_thread_num();
            tmp_hist[tid][val].c++;
        }
        free_rand(rg);
    }

    #pragma omp parallel for 
    for (int i = 0; i < num_buckets; i++) {
        for (int tid = 0; tid < num_threads; tid++) {
            histogram[i] += tmp_hist[tid][i].c;
        }
    }
    return 0;
    // naive parallelization
    // volatile int *histogram = (int*) calloc(num_buckets, sizeof(int));
    // omp_set_num_threads(num_threads);
    // #pragma omp parallel
    // {
    //     rand_gen generator = init_rand();
    //     #pragma omp for 
    //     for(int i = 0; i < num_samples; i++){
    //         int val = next_rand(generator) * num_buckets;
    //         #pragma omp critical
    //         histogram[val]++;
    //     }
    //     free_rand(generator);
    // }
    // return 0;
}
