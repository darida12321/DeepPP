CC = g++
CFLAGS = -Wall -O3 -std=c++20

.PHONY: all
all: bin/layer.o bin/network.o

.PHONY: test
test: bin/test_layer

bin:
	mkdir bin

bin/layer.o: include/layer.h src/layer.cc | bin
	$(CC) $(CFLAGS) -c src/layer.cc -o bin/layer.o

bin/network.o: include/network.h src/network.cc bin/layer.o | bin
	$(CC) $(CFLAGS) -c src/network.cc -o bin/network.o
	
bin/test_layer: bin/layer.o bin/network.o | bin
	$(CC) $(CFLAGS) test/test_layer.cc bin/layer.o bin/network.o -o bin/test_layer
	./bin/test_layer


.PHONY: clean
clean:
	rm -rf bin/
