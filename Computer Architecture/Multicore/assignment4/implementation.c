/*
============================================================================
Filename    : algorithm.c
Author      : Your name goes here
SCIPER      : Your SCIPER number
============================================================================
*/

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}


__global__ void gpu_process(double *input, double *output, int length) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int center = (i == length/2-1 && j == length/2-1) ||
                 (i == length/2 && j == length/2-1) ||
                 (i == length/2-1 && j == length/2) ||
                 (i == length/2 && j == length/2);
    if (i <= 0 || j <= 0 || i >= length-1 || j >= length-1 || center) 
        return;
    output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                input[(i-1)*(length)+(j)]   +
                                input[(i-1)*(length)+(j+1)] +
                                input[(i)*(length)+(j-1)]   +
                                input[(i)*(length)+(j)]     +
                                input[(i)*(length)+(j+1)]   +
                                input[(i+1)*(length)+(j-1)] +
                                input[(i+1)*(length)+(j)]   +
                                input[(i+1)*(length)+(j+1)] ) / 9;
}

__global__ void gpu_process_shared(double *input, double *output, int length) {
    __shared__ double temp[100];
    int slength = 10;
    int temp_i = threadIdx.x + 1;
    int temp_j = threadIdx.y + 1;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int center = (i == length/2-1 && j == length/2-1) ||
                 (i == length/2 && j == length/2-1) ||
                 (i == length/2-1 && j == length/2) ||
                 (i == length/2 && j == length/2);
    
    if (i <= 0 || j <= 0 || i >= length-1 || j >= length-1) 
        return;
    temp[(temp_i)*(slength)+(temp_j)] = input[(i)*(length)+(j)];
    if (center) 
        return;
    if (temp_i == 1 || i == 1) {
        temp[(temp_i-1)*(slength)+(temp_j-1)] = input[(i-1)*(length)+(j-1)];
        temp[(temp_i-1)*(slength)+(temp_j)]   = input[(i-1)*(length)+(j)];
        temp[(temp_i-1)*(slength)+(temp_j+1)] = input[(i-1)*(length)+(j+1)];
    } 
    if (temp_j == 1 || j == 1) {
        temp[(temp_i)*(slength)+(temp_j-1)]   = input[(i)*(length)+(j-1)];
    }
    if (temp_i == 8 || i == length-2) {
        temp[(temp_i+1)*(slength)+(temp_j-1)] = input[(i+1)*(length)+(j-1)];
        temp[(temp_i+1)*(slength)+(temp_j)]   = input[(i+1)*(length)+(j)];
        temp[(temp_i+1)*(slength)+(temp_j+1)] = input[(i+1)*(length)+(j+1)];
    } 
    if (temp_j == 8 || j == length-2) {
        temp[(temp_i)*(slength)+(temp_j+1)]   = input[(i)*(length)+(j+1)];
    }
    
    __syncthreads();
    output[(i)*(length)+(j)] = (temp[(temp_i-1)*(slength)+(temp_j-1)] +
                                temp[(temp_i-1)*(slength)+(temp_j)]   +
                                temp[(temp_i-1)*(slength)+(temp_j+1)] +
                                temp[(temp_i)*(slength)+(temp_j-1)]   +
                                temp[(temp_i)*(slength)+(temp_j)]     +
                                temp[(temp_i)*(slength)+(temp_j+1)]   +
                                temp[(temp_i+1)*(slength)+(temp_j-1)] +
                                temp[(temp_i+1)*(slength)+(temp_j)]   +
                                temp[(temp_i+1)*(slength)+(temp_j+1)] ) / 9;
}
__global__ void update_center(double* output, int length) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    output[(i+length/2-1)*length+(j+length/2-1)] = 1000;
}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);


    /* Preprocessing goes here */
    int size = length * length * sizeof(double);
    double *cuda_input, *cuda_output, *temp;
    // The link suggests threads per block >= 128
    // but seems a bit too much for this assignment
    // https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529
    // int threadsPerBlock = 64;
    int dimBlock = 8;
    // equivalent to ceil
    int dimGrid = (length + dimBlock - 1) / dimBlock;
    dim3 block(dimBlock, dimBlock);
    dim3 grid(dimGrid, dimGrid);
    dim3 centerdim(2, 2);
    cudaMalloc((void**)&cuda_input, size);
    cudaMalloc((void**)&cuda_output, size);

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    cudaMemcpy(cuda_input, input, size, cudaMemcpyHostToDevice);
    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    cudaEventRecord(comp_start);
    /* GPU calculation goes here */
    update_center <<< 1, centerdim>>> (cuda_output, length);
    for (int i = 0; i < iterations; i++) {
        gpu_process_shared <<< grid, block >>> (cuda_input, cuda_output, length);
        temp = cuda_input;
        cuda_input = cuda_output;
        cuda_output = temp;
    }
    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
    cudaMemcpy(output, cuda_input, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(cuda_input);
    cudaFree(cuda_output);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}