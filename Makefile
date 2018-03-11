CFLAGS=-O3 -ggdb3 --std=gnu11 -Wall
CXXFLAGS=-O3 -ggdb3 -Wall

default: beatthehash.cu
	nvcc $< -o beatthehash

clean:

.PHONY: clean
