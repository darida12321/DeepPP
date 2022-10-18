#! /bin/sh
git submodule init
git submodule update
cmake -S . -B ./bin
