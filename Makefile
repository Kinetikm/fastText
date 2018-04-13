#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x -march=core2
INCLUDES = -I.
LIBNAME = fastText
LIBOBJS = args.o dictionary.o matrix.o vector.o utils.o fasttext.o main.o

opt: CXXFLAGS += -O3 -funroll-loops
opt: $(LIBNAME).a

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

main.o: src/main.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/main.cc

$(LIBNAME).a: $(LIBOBJS)
	 ar rcs $(LIBNAME).a $^

clean:
	rm -rf *.o fasttext $(LIBNAME).a
