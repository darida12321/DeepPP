CC = g++
CFLAGS = -c -Wall -O3

.PHONY: all
all: bin/layer

bin:
	mkdir bin

bin/layer: src/layer.h src/layer.cc | bin
	$(CC) $(CFLAGS) src/layer.cc -o bin/layer

.PHONY: clean
clean:
	rm -rf bin/
