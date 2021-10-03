#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
typedef unsigned long long bignum;

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void computePrimes(char results[], bignum s, bignum n){
   
    bignum i;

    if(s % 2 == 0) s ++;  //make sure s is an odd number
 
    for(i=s; i< s+n; i = i + 2){
       results[i]=isPrime(i);
    }
 }

 __global__ void elementPrime(bignum *a, char *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

int isPrime(bignum x){

    bignum i;
    bignum lim = (bignum) sqrt(x) + 1;
       
    for(i=2; i<lim; i++){
       if ( x % i == 0)
          return 0;
    }
 
    return 1;
 }
 
 
void initializeArray(char a[], bignum len){
 // init results char array
    int i;
    
    for(i=0; i<len; i++){
       a[i]= 0;
    }
 
 }
 
void printArray(char a[], int len){
 
    int i;
    
    for(i=0; i<len; i++){
    
       printf("%d ", a[i]);
 
    }
 
 }
 
int arrSum( char a[], bignum len )
 {
     int i, s = 0;
     for( i = 0; i < len; i ++ )
         s += a[i];
 
     return s;
 }

 
int main( int argc, char* argv[] )
{
    
    if(argc < 3)
    {
        printf("Usage: too few arguments\n");
        exit(-1);
    }
    // Retrieve N, blockSize from args
    bignum N = (bignum) atoi(argv[1]);
    bignum blockSize = (bignum) atoi(argv[2])

    // Size, in bytes, of each vector
    size_t bytes = N*sizeof(bignum);
    
    // Host vectors
    bignum *h_a;
    char *h_results;

    // Device vectors
    bignum *d_a;
    char *d_results;
    
    // Allocate for host
    h_a = (bignum*)malloc(bytes);
    h_results = (char*)malloc((N + 1) * sizeof(char));

    // Allocate for device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_results, (N + 1) * sizeof(char));

    // Init timing vars
    double now_cpu, then_cpu;
    double now_gpu, then_gpu;
    double cost_cpu, cost_gpu;
  

    free(h_a);
    free(h_results);

    cudaFree(d_a);
    cudaFree(d_results);
    // int i;
    // // Initialize vector on host
    // for( i = 3; i < N; i++ ) {
    //     h_a[i] = i;
    //     h_results[i] = 0
    // }
 
    // // Copy host vector to device
    // cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
 
    // int blockSize, gridSize;
 
    // // Number of threads in each thread block
    // blockSize = 1024;
 
    // // Number of thread blocks in grid
    // gridSize = (int)ceil((float)n/blockSize);
 
    // // Execute the kernel
    // vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // // Copy array back to host
    // cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // // Sum up vector c and print result divided by n, this should equal 1 without error
    // double sum = 0;
    // for(i=0; i<n; i++)
    //     sum += h_c[i];
    // printf("final result: %f\n", sum/n);
 
    // // Release device memory
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);
 
    // // Release host memory
    // free(h_a);
    // free(h_b);
    // free(h_c);
    
    // bignum *a = malloc(N *sizeof(bignum));
    // char *results = malloc((N + 1) * sizeof(char));
    // double now, then;
    // double scost, pcost;
       
    // initializeArray(results, N);
    // printf("%%%%%% Find all prime numbers in the range of 3 to %llu.\n", N);   
  
    // then = currentTime();
    // h_computePrimes(results, 3, N);
    // now = currentTime();
    // scost = now - then;
    // printf("%%%%%% Serial code execution time in second is %lf\n", scost);
 
 
    // printf("Total number of primes in that range is: %d.\n\n", arrSum(results, N + 1));
    printf("Cool Beans\n");
    return 0;
}
