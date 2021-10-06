#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__device__ int isPrimeGPU(long x) {

    long long i;
    for (i = 2; i * i < x + 1; i++) {
        if (x % i == 0) {
            return 0;
        }
    }
    return 1;
}

__host__ int isPrime(long x) {

    long i;
    for (i = 2; i < sqrt(x) + 1; i++) {
        if (x % i == 0) {
            return 0;
        }
    }
    return 1;
}


__global__ void primeFind(int* c, long n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    c[0] = (int)0;
    c[1] = (int)0;
    c[2] = (int)1;
    long num = (id * 2)-1;
    
    if (id < ((n/2)+1) && 1 < id) {//global thread 0 and 1 dont do anything need to deal with         
        if (id * 2 < n) {//even numbers
            c[id * 2] = 0;
        }
        if (num < n) {//odd numbers
            c[num] = isPrimeGPU(num);
        }
    }
    
}





int main(int argc, const char* argv[]) {
    
    if (argc < 3){
        printf("Usage: prime upbound\n");
        exit(-1);
    }
    // Size of vectors
    long n = atoi(argv[1]);
    printf("n = %ld \n", n);

    if(n <= 0){
        printf("Usage: prime upbound, you input invalid upbound number!\n");
        exit(-1);
    }

    int blockSize = atoi(argv[2]);
    printf("block size = %d \n", blockSize);


    
    // Host output 
    int* cpuOutput;
    //Device output vector
    int* gpuOutput;


    // Size, in bytes, of each output
    size_t bytes = (unsigned long long)n * sizeof(int);
    // Allocate memory for each vector on host
    cpuOutput = (int*)malloc(bytes);//pc results new int[n]
    gpuOutput = (int*)malloc(bytes);//gpu results
    
    //initalize
    for (long j = 0; j < n; j++) {
        cpuOutput[j] = 0;
        gpuOutput[j] = 0;
    }

    clock_t cStart = clock();
    double cpuStart = (double) cStart/CLOCKS_PER_SEC;// 
    ///////////////////////////////////////////////////////////////////////////////////
    //do it on cpu
    //TODO add systime to check how long it takes
    
    cpuOutput[0] = (int)0;
    cpuOutput[1] = (int)0;
    cpuOutput[2] = (int)1;
    for (long i = 2; i < (n/2)+1; i++) {
        long num = (i * 2) - 1;
        if (i * 2 < n) {
            cpuOutput[i * 2] = 0;
        }
        if (num < n) {
            cpuOutput[num] = isPrime(num);
        }
    }
    
    clock_t cEnd = clock();
    double cpuEnd = (double)cEnd/CLOCKS_PER_SEC;
    //sum up pc result of # of primes
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (int)cpuOutput[i];
    }
    printf("CPU final result: %d\n", sum);

    
    double cpuTotal = cpuEnd - cpuStart;
    printf("CPU took %lf seconds to find primes numbers up to %ld\n", cpuTotal, n);

    ////////////////////////////////////////////////////////////////////////
    //do it on gpu
    //TODO sys clock time for seeing how much time it takes

    clock_t gStart = clock();
    double gpuStart = (double)gStart / CLOCKS_PER_SEC;;
    //Device output vector
    int* d_output;
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_output, bytes);


    // Copy host vectors to device//i dont think we need to do this
    cudaMemcpy(d_output, gpuOutput, bytes, cudaMemcpyHostToDevice);

    int gridSize;
    // Number of thread blocks in grid
    gridSize = (int)ceil((double)((double)((n+1)/2)/blockSize));
    
    
    primeFind<<<gridSize, blockSize>>>(d_output, n);

    // Copy array back to host
    cudaMemcpy(gpuOutput, d_output, bytes, cudaMemcpyDeviceToHost);

    clock_t gEnd = clock();
    double gpuEnd = (double)gEnd / CLOCKS_PER_SEC;
    // Sum up vector c and print result divided by n, this should equal 1 without error
    sum = 0;
    for (long i = 2; i < n; i++) {
        sum += (int)gpuOutput[i];
    }
    printf("GPU final result: %d\n", sum);
    
    
    long double gpuTotal = gpuEnd - gpuStart;

    printf("GPU took %Lf seconds to find primes numbers up to %ld\n", gpuTotal, n);

    printf("GPU speeds up the process %Lf times.\n", cpuTotal / gpuTotal);

    // Release device memory
    cudaFree(d_output);

    // Release host memory
    free(cpuOutput);
    free(gpuOutput);

    return 0;
}







