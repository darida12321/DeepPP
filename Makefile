CC = g++
# CFLAGS1 = -c -Wall -O3
CFLAGS = -Wall -O3

.PHONY: all
all: bin/layer bin/network bin/main

bin:
	mkdir bin

bin/layer: src/layer.h src/layer.cc | bin
	g++ -c src/layer.cc -o bin/layer.o
	# $(CC) $(CFLAGS) -c src/layer.cc -o bin/layer.o

bin/network: src/network.h src/network.cc bin/layer | bin
	g++ -c src/network.cc -o bin/network.o
	# $(CC) $(CFLAGS) -c src/network.cc -o bin/network.o

bin/main: src/main.cc bin/network | bin
	g++ -c src/main.cc -o bin/main.o
	g++ bin/network.o bin/main.o -o bin/main
	# $(CC) $(CFLAGS) -c src/main.cc -o bin/main.o

.PHONY: clean
clean:
	rm -rf bin/
