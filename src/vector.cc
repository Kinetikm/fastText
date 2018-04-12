/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include <assert.h>
#include <iomanip>
#include <cmath>
#include "matrix.h"

namespace fasttext {

Vector::Vector(int64_t m) : data_(m) {}

void Vector::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

real Vector::norm() const {
  real sum = 0;
  for (int64_t i = 0; i < size(); i++) {
    sum += data_[i] * data_[i];
  }
  return std::sqrt(sum);
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < size(); i++) {
    data_[i] *= a;
  }
}

void Vector::addVector(const Vector& source) {
  assert(size() == source.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] += source.data_[i];
  }
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  for (int64_t j = 0; j < A.size(1); j++) {
    data_[j] += A.at(i, j);
  }
}
std::ostream& operator<<(std::ostream& os, const Vector& v)
{
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.size(); j++) {
    os << v[j] << ' ';
  }
  return os;
}

}
