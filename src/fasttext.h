/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <time.h>

#include <atomic>
#include <memory>
#include <set>
#include <chrono>
#include <iostream>
#include <queue>
#include <tuple>
#include <map>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
 protected:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;

  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;
  std::map<std::string, double> idfs_mapping_;
  bool idfs_weights_ = false;
  int32_t version;
  bool checkModel(std::istream&);

 public:
  FastText();

  void getSubwordVector(Vector&, const std::string&) const;
  void addInputVector(Vector&, int32_t) const;
  void getWordVector(Vector&, const std::string&) const;
  void saveOutput();
  void loadModel(std::istream&);
  void loadModel(const std::string&);
  void loadModel(const std::string&, const std::string&);
  int getDimension() const;
  void getSentenceVector(std::string sentence, Vector&);
};
}
