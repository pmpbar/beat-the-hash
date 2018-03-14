TARGET=beatthehash

CFLAGS=-O3 -ggdb3 --std=gnu11 -Wall
CXXFLAGS=-O3 -ggdb3 -Wall


# threefish: src/threefish.cu
	# ${CC} ${CFLAGS} -o build/$@ $<

default: src/beatthehash.cu src/threefish.cu
	nvcc $< -o build/$(TARGET) -Iinc

clean:
	rm -rf $(TARGET)

.PHONY: clean
