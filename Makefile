CC = g++
# CFLAGS1 = -c -Wall -O3
CFLAGS = -Wall -O3

.PHONY: all
all: bin/layer.o bin/network.o

bin:
	mkdir bin

bin/layer.o: include/layer.h src/layer.cc | bin
	g++ -c src/layer.cc -o bin/layer.o

bin/network.o: include/network.h src/network.cc bin/layer.o | bin
	g++ -c src/network.cc -o bin/network.o
	
bin/test_layer: bin/layer.o bin/network.o | bin
	g++ test/test_layer.cc bin/layer.o bin/network.o -o bin/test_layer


.PHONY: clean
clean:
	rm -rf bin/
