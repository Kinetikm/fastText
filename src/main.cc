/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>
#include <queue>
#include <iomanip>
#include "fasttext.h"
#include "args.h"

using namespace fasttext;


int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 1) {
    return 1;
  }
  FastText fasttext;
  fasttext.loadModel(args[1]);
  Vector svec(fasttext.getDimension());
  std::string sentence;
  std::cin >> sentence;
  fasttext.getSentenceVector(sentence, svec);
  std::cout << svec << std::endl;
  
  return 0;

  }
