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
    
    printArray(results, n);
    for(i=s; i< s+n; i = i + 2){
        
        results[i]=h_isPrime(i);
        printf("here %llu [%c]\n", i, results[i]);
    }
 }

 __global__ void elementPrime(bignum *a, char *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(*a % 2 == 0);  //make sure a is an odd number

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = d_isPrime(*a);

}

__device__ int d_isPrime(bignum x){

    bignum i;
    // bignum lim = (bignum) sqrt(x) + 1;
       
    for(i=2; (i-1)*(i-1)<x; i = i + 2){
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
    int blockSize = (int) atoi(argv[2]);

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

    // Init timing vars
    double now_cpu, then_cpu;
    double now_gpu, then_gpu;
    double cost_cpu, cost_gpu;
  
    // Allocate for CPU proc
    char *results = (char *)malloc((N + 1) * sizeof(char));
    printArray(results, N);

    int i;
    // Initialize vector on host
    for( i = 3; i < N; i++ ) {
        h_a[i] = i;
        h_results[i] = 0;
        results[i] = 0;
    }
 
    then_gpu = currentTime();

    // Allocate for device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_results, (N + 1) * sizeof(char));

    // Copy host vector to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
 
    // Number of thread blocks in grid
    int gridSize;
    gridSize = (int)ceil((float)N/blockSize);
 
    // Execute the kernel
    elementPrime<<<gridSize, blockSize>>>(d_a, d_results, N);
 
    // Copy array back to host
    cudaMemcpy( h_results, d_results, bytes, cudaMemcpyDeviceToHost );
 
    now_gpu = currentTime();
    cost_gpu = now_gpu - then_gpu;
       
    initializeArray(results, N);
    printf("%%%%%% Find all prime numbers in the range of 3 to %llu.\n", N);   
  
    then_cpu = currentTime();
    computePrimes_cpu(results, 3, N - 3);
    now_cpu = currentTime();
    cost_cpu = now_cpu - then_cpu;
    printf("%%%%%% Serial code execution time in second is %lf\n", cost_cpu);
    printf("%%%%%% Paralel code execution time in second is %lf\n", cost_gpu);
 
 
    printf("CPU: Total number of primes in that range is: %d.\n\n", arrSum(h_results, N + 1));
    printf("GPU: Total number of primes in that range is: %d.\n\n", arrSum(results, N + 1));
    printf("Cool Beans\n");

    free(h_a);
    free(h_results);
    free(results);

    cudaFree(d_a);
    cudaFree(d_results);

    return 0;
}
