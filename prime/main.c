#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include "prime.h"


int main (int argc, const char * argv[]) {    
    // arc is the number of args, argv is the vector of args passed in.
    // argv[1] is the name of the .exe

   if(argc < 2)
   {
       printf("Usage: prime upbound\n");
       exit(-1);
   }
   // typecasting to bignum type (64bit) from str arg
   // atoi convert str to int
   bignum N = (bignum) atoi(argv[1]);
   if(N <= 0)
   {
       printf("Usage: prime upbound, you input invalid upbound number!\n");
       exit(-1);
   }
   
   // make bignum array a with size N
   bignum *a = malloc(N *sizeof(bignum));
   // make char array with results 
   char *results = malloc(N * sizeof(char));
   double now, then;
   double scost, pcost;
    
   // copy numbers 0-N into a
   initializeArray(a, N);
   // long long in str -> %llu, long -> %lu, double/float -> %lf
   printf("%%%%%% Find all prime numbers in the range of 0 to %llu.\n", N);   
 
   then = currentTime();
   computePrimes(results, 0, N);
   now = currentTime();
   scost = now - then;
   printf("%%%%%% Serial code executiontime in second is %lf\n", scost);

//    then = currentTime();
//    pcomputePrimes(results, 0, N);
//    now = currentTime();
//    pcost = now - then;
//    printf("%%%%%% Parallel code executiontime with 4 threads in second is %lf\n", pcost);
   
//    printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using 4 threads is %lf\n", scost / pcost); 
//    printf("%%%%%% The efficiency(Speedup / NumProcessorCores) when using 4 threads is %lf\n", scost / pcost / 4); 

   printArray(results, N);
   return 0;
}
