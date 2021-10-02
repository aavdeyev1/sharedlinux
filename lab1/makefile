CC=nvcc
CFLAGS=-I.
DEPS = timing.h prime.h

%.o: %.c $(DEPS)
	$(CC) -O2 -c -o $@ $< $(CFLAGS)

lab1: main.o timing.o prime.o
			nvcc -arch=sm_30 -o lab1 timing.o main.o prime.o

main.o: main.cu
			nvcc -arch=sm_30 -c main.cu

prime.o: prime.c prime.h
			g++ -c -x c++ timing.c -I.

timing.o: timing.c timing.h
			g++ -c -x c++ timing.c -I.

clean:
			rm -r *.o lab1