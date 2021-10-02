CC=gcc
CFLAGS=-I.
DEPS = timing.h prime.h 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

findPrime: timing.o prime.o main.o
	gcc -o findPrime timing.o prime.o main.o -I. -lm -lpthread

clean:
	rm -r *.o findPrime
