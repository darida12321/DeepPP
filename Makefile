CC = g++
CFLAGS1 = -c -Wall -O3
CFLAGS2 = -Wall -O3

.PHONY: all
all: bin/layer bin/network bin/main

bin:
	mkdir bin

bin/layer: src/layer.h src/layer.cc | bin
	$(CC) $(CFLAGS1) src/layer.cc -o bin/layer

bin/network: src/network.h src/network.cc | bin
	$(CC) $(CFLAGS1) src/network.cc -o bin/network

bin/main: src/main.cc | bin
	$(CC) $(CFLAGS2) src/main.cc -o bin/main

.PHONY: clean
clean:
	rm -rf bin/
