/*
============================================================================
Filename    : pi.c
Author      : Your names goes here
SCIPER		: Your SCIPER numbers
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "utility.h"

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


void* inCircle(void* res) {
    int tid = pthread_self();
    rand_gen rand = init_rand_pthreads(tid);
    int i = *(int*) res;
    for (; i >0; i--) {
        double x = next_rand(rand);
        double y = next_rand(rand);
        if (x*x + y*y > 1) {
            *(int*)res -= 1;
        }
    }
    free_rand(rand);
    return NULL;
}

double calculate_pi (int num_threads, int samples) {
    double pi;
    pthread_t threads[num_threads];
    int res[num_threads];
    int sum = 0;
    /* Your code goes here */
    for (int i=0; i<num_threads; i++) {
        res[i] = samples / num_threads;
        pthread_create(&threads[i], NULL, inCircle, (void*) (res + i));
    }
    for (int i=0; i<num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    for (int i=0; i<num_threads; i++) {
        sum += res[i];
    }
    pi = 4.0 * sum / samples;
    return pi;
}
