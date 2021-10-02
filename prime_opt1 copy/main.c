#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include "prime.h"


int main (int argc, const char * argv[]) {    
   
   if(argc < 2)
   {
       printf("Usage: prime upbound\n");
       exit(-1);
   }
   bignum N = (bignum) atoi(argv[1]);
   if(N <= 0)
   {
       printf("Usage: prime upbound, you input invalid upbound number!\n");
       exit(-1);
   }
   
   bignum *a = malloc(N *sizeof(bignum));
   char *results = malloc((N + 1) * sizeof(char));
   double now, then;
   double scost, pcost;
      
   initializeArray(results, N);
   printf("%%%%%% Find all prime numbers in the range of 0 to %llu.\n", N);   
 
   then = currentTime();
   computePrimes(results, 0, N);
   now = currentTime();
   scost = now - then;
   printf("%%%%%% Serial code executiontime in second is %lf\n", scost);
   printf("Total number of primes in that range is: %d.\n\n", arrSum(results, N + 1));

   initializeArray(results, N);
   then = currentTime();
   pcomputePrimes(results, 0, N);
   now = currentTime();
   pcost = now - then;
   printf("%%%%%% Parallel code executiontime with 4 threads in second is %lf\n", pcost);
   printf("Total number of primes in that range is: %d.\n\n", arrSum(results, N + 1));
   
   printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using 4 threads is %lf\n", scost / pcost); 
   printf("%%%%%% The efficiency(Speedup / NumProcessorCores) when using 4 threads is %lf\n", scost / pcost / 4); 

   printArray(results, N);
   return 0;
}
