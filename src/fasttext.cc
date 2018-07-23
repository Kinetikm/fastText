#include "fasttext.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>

#include <stdlib.h>


namespace fasttext {

constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

FastText::FastText() {}

void FastText::addInputVector(Vector& vec, int32_t ind) const {
    vec.addRow(*input_, ind);
}


void FastText::getSubwordVector(Vector& vec, const std::string& subword)
    const {
  vec.zero();
  int32_t h = dict_->hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void FastText::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  vec.zero();
  for (int i = 0; i < ngrams.size(); i ++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::saveOutput() {
  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        args_->output + ".output" + " cannot be opened for saving vectors!");
  }
  int32_t n = (args_->model == model_name::sup) ? dict_->nlabels()
                                                : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                         : dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(const std::string& filename, const std::string& idfs_filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
  std::string line;
  std::ifstream idfIfs(idfs_filename);
  if (!idfIfs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  idfs_weights_ = true;
  idfs_mapping_ = std::map<std::string, double>;
  while(idfIfs>>line){
      std::stringstream ss(line);
      std::string item;
      std::vector<std::string> splits;
      while (std::getline(ss, item, ' ')) {
          *(splits++) = item;
      }
      if (splits.size()<2){
          continue;
      }
      idfs_mapping_[splits[0]] = strtod((splits[1]).c_str(),0);
  }
  idfIfs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  args_->load(in);
  dict_ = std::make_shared<Dictionary>(args_, in);
  bool quant_input;
  in.read((char*) &quant_input, sizeof(bool));
  input_->load(in);
  in.read((char*) &args_->qout, sizeof(bool));
  output_->load(in);
}

void FastText::getSentenceVector(
    std::string sentence,
    fasttext::Vector& svec) {
  svec.zero();
  Vector vec(args_->dim);
  std::istringstream iss(sentence);
  int32_t count = 0;
  std::string word;
  while (iss >> word) {
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      vec.mul(1.0 / norm);
      svec.addVector(vec);
      count++;
    }
  }
  if (count > 0) {
    svec.mul(1.0 / count); // <<<
  }
}

int FastText::getDimension() const {
    return args_->dim;
}

}
