#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
 
typedef unsigned long long bignum;

void initializeArray(char a[], bignum len);
__device__ int d_isPrime(bignum x);
void computePrimes_cpu(char results[], bignum s, bignum n);
int arrSum(char results[], bignum len);
__host__ int h_isPrime(bignum x);
__host__ double currentTime();
void printArray(char a[], int len);


__host__ double currentTime()
{

   struct timeval now;
   gettimeofday(&now, NULL);
   
   return now.tv_sec + now.tv_usec/1000000.0;
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void computePrimes_cpu(char results[], bignum s, bignum n){
   
    bignum i;

    if(s % 2 == 0) s ++;  //make sure s is an odd number
    
    // printArray(results, n);
    for(i=s; i< n; i = i + 2){
        
        results[i]=h_isPrime(i);
        // printf("here %llu [%d]\n", i, results[i]);
    }
 }

 __global__ void elementPrime(bignum *a, char *results, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    
    // if(*a == 0 | *a == 1 | *a == 2 or *a == 3);
    // if(id % 2 == 0);  //make sure a is an odd number

    // Make sure we do not go out of bounds
    
    if (id < n)
    {
        results[id] = d_isPrime(*a);
    }
}

__device__ int d_isPrime(bignum x){

    bignum i;
    // bignum lim = (bignum) sqrt(x) + 1;
       
    for(i=2; i*i<x; i = i++){
       if ( x % i == 0)
          return 0;
    }
 
    return 1;
 }
 
 __host__ int h_isPrime(bignum x){

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
 
 void initializeArray_gpu(bignum a[], bignum len){
    // init results char array
       bignum i;
       
       for(i=3; i< len; i = i + 2)
       {
            a[i] = i;
            printf("here %llu [%llu]\n", i, a[i]);
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
     {
         s += a[i];
         printf(" | %d -> %d", i, a[i]);
     }
 
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
    int blockSize = (int) atoi(argv[2]);

    // Size, in bytes, of each vector
    size_t bytes = ceil((float)N/2);
    
    // Host vectors
    bignum *h_a;
    char *h_results;

    // Device vectors
    bignum *d_a;
    char *d_results;
    
    // Allocate for host
    h_a = (bignum*)malloc(bytes);
    h_results = (char*)malloc((N + 1) * sizeof(char));

    // Init timing vars
    double now_cpu, then_cpu;
    double cost_cpu;

    double now_gpu, then_gpu;
    double cost_gpu;
  
    
    // h_results[0] = 0;
    // h_a[0] = 0;
    // h_results[1] = 0;
    // h_a[1] = 1;
    // h_results[2] = 1;
    // h_a[2] = 2;
    // h_results[3] = 1;
    // h_a[3] = 3;

    // bignum i;
    // Initialize vector on host
    // for( i = 4; i < N + 1; i++ ) {
    //     h_a[i] = i;
    //     h_results[i] = 0;
    //     // results[i] = 0;
    // }
    // printArray(h_results, N - 3);

    bignum len_a = ceil((float)N/2);
    initializeArray_gpu(h_a, N);
    printf("\n%%%%%% GPU: Find all prime numbers in the range of 3 to %llu.\n", len_a);   

    printf("GPU ARRAY 1.0\n");
    for (int j=0; j < len_a; j++){
        printf("%llu ", h_a[j]);
    }
    then_gpu = currentTime();

    // Allocate for device
    cudaMalloc(&d_a, len_a);
    cudaMalloc(&d_results, (N + 1) * sizeof(char));

    // Copy host vector to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
 
    // Number of thread blocks in grid
    int gridSize;
    gridSize = (int)ceil((float)N/blockSize);
 
    // Execute the kernel
    elementPrime<<<gridSize, blockSize>>>(d_a, d_results, bytes);
 
    // Copy array back to host
    cudaMemcpy( h_results, d_results, bytes, cudaMemcpyDeviceToHost );
    printf("GPU ARRAY 2.0\n");
    printArray(h_results, N + 1);
 
    now_gpu = currentTime();
    cost_gpu = now_gpu - then_gpu;

    // print output GPU
    printf("%%%%%% Parallel code execution time in second is %lf\n", cost_gpu);
    printf("GPU: Total number of primes in that range is: %d.\n\n", arrSum(h_results, N + 1));




    // Allocate for CPU proc
    // char *results = (char*)malloc((N + 1) * sizeof(char));
    // printArray(h_results, N);   

    
  
    then_cpu = currentTime();

    

    // bignum i;
    // bignum s = 3;
    // for(i=s; i< s+N; i++){
   
    //     results[i]=h_isPrime(i);
    // }
    initializeArray(h_results, N);
    h_results[0] = 1;
    h_results[1] = 1;
    h_results[2] = 1;
    h_results[3] = 1;
    computePrimes_cpu(h_results, 3, N);
    // printf("CPU ARRAY 2.0\n");
    // printArray(h_results, N);
    now_cpu = currentTime();
    cost_cpu = now_cpu - then_cpu;
    printf("%%%%%% Serial code execution time in second is %lf\n", cost_cpu);
 
    printf("CPU: Total number of primes in that range is: %d.\n\n", arrSum(h_results, N + 1));
    
    printf("Cool Beans\n");

    free(h_a);
    free(h_results);
    // free(results);

    cudaFree(d_a);
    cudaFree(d_results);

    return 0;
}
