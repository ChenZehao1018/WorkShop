/*
============================================================================
Filename    : algorithm.c
Author      : Your names go here
SCIPER      : Your SCIPER numbers

============================================================================
*/
#include <math.h>
#include <omp.h>
#include "utility.h"

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    double *temp;
    omp_set_num_threads(threads);
    // Parallelize this!!
    for(int n=0; n < iterations; n++)
    {
        #pragma omp parallel
        {
            #pragma omp for
            for(int i=1; i<length-1; i++)
            {   
                for(int j=1; j<length-1; j++)
                {
                    if ( ((i == length/2-1) || (i== length/2))
                        && ((j == length/2-1) || (j == length/2)) )
                        continue;

                    OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                                INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                                INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
                }
                // loop unrolling attempt, no observable improvement
                // for(int j=1; j<length-1; j+=2)
                // {
                //     if ( ((i == length/2-1) || (i== length/2))
                //         && ((j == length/2-1) || (j == length/2)) )
                //         continue;
                //     OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                //                 INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                //                 INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
                //     j++;
                //     if ( ((i == length/2-1) || (i== length/2))
                //         && ((j == length/2-1) || (j == length/2)) )
                //         continue;
                //     OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                //                 INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                //                 INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
                //     j--;
                // }
            }
            // for timing 
            // struct timeval t;
            // gettimeofday(&t, NULL);
            // double elapsed = (t.tv_sec - start.tv_sec); 
            // elapsed += (double)(t.tv_usec - start.tv_usec) / 1000000.0;
            // printf("Thread %d finished in %.4gs.\n", omp_get_thread_num(), elapsed);
        }

        temp = input;
        input = output;
        output = temp;
    }
}
