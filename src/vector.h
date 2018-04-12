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
#include <ostream>
#include <vector>

#include "real.h"

namespace fasttext {

class Matrix;

class Vector {

  protected:
    std::vector<real> data_;

  public:
    explicit Vector(int64_t);
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    inline real* data() {
      return data_.data();
    }
    inline const real* data() const {
      return data_.data();
    }
    inline real& operator[](int64_t i) {
      return data_[i];
    }
    inline const real& operator[](int64_t i) const {
      return data_[i];
    }

    inline int64_t size() const {
      return data_.size();
    }
    void mul(real);
    void zero();
    real norm() const;
    void addVector(const Vector& source);
    void addRow(const Matrix&, int64_t);
};

std::ostream& operator<<(std::ostream&, const Vector&);

}
