CC=gcc
CFLAGS=-I.
DEPS = timing.h prime.h 

%.o: %.c $(DEPS)
	$(CC) -O2 -c -o $@ $< $(CFLAGS)

findPrime: timing.o prime.o main.o
	gcc -o findPrime timing.o prime.o main.o -I. -O2 -lm -lpthread

clean:
	rm -r *.o findPrime
