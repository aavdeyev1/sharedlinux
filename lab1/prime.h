#define NTHREADS 4

typedef unsigned long long bignum;

typedef struct{
   char *results;
   bignum start;
   bignum max;
   int tid;
} jobSpec;

void initializeArray(char a[], bignum len);
int isPrime(bignum x);
void printArray(char a[], int len);
void computePrimes(char results[], bignum s, bignum n);
int pcomputePrimes(char results[], bignum s, bignum n);
void * primeThread(void *args);
int arrSum(char results[], bignum len);
