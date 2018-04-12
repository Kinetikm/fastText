/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <assert.h>
#include "real.h"

namespace fasttext {

class Vector;

class Matrix {
 protected:
  std::vector<real> data_;
  const int64_t m_;
  const int64_t n_;

 public:
  Matrix();
  explicit Matrix(int64_t, int64_t);
  Matrix(const Matrix&) = default;
  Matrix& operator=(const Matrix&) = delete;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }

  inline const real& at(int64_t i, int64_t j) const {
    return data_[i * n_ + j];
  };
  inline real& at(int64_t i, int64_t j) {
    return data_[i * n_ + j];
  };

  inline int64_t size(int64_t dim) const {
    assert(dim == 0 || dim == 1);
    if (dim == 0) {
      return m_;
    }
    return n_;
  }
  inline int64_t rows() const {
    return m_;
  }
  inline int64_t cols() const {
    return n_;
  }
  void zero();
  void load(std::istream&);

  void dump(std::ostream&) const;
};
}
