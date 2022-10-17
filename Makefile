CC = g++
CFLAGS = -c -Wall -O3

.PHONY: all
all: bin/layer

bin:
	mkdir bin

bin/layer: layer.cc | bin
	$(CC) $(CFLAGS) layer.cc -o bin/layer

.PHONY: clean
clean:
	rm -rf bin/
