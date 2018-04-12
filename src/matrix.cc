/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "matrix.h"

#include <random>
#include <exception>
#include <stdexcept>
#include "utils.h"
#include "vector.h"

namespace fasttext {

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(int64_t m, int64_t n) : data_(m * n), m_(m), n_(n) {}

void Matrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void Matrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = std::vector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

}
