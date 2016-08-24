#include "mnist.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

static int reverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

std::unique_ptr<unsigned[]> readMNISTLabels(int numberOfImages,
                                            const char *fname) {
  auto arr =
    std::unique_ptr<unsigned[]>(new unsigned[numberOfImages]);
  std::ifstream file(fname,std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*) &number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    for(int i = 0; i < number_of_images; ++i) {
      unsigned char temp = 0;
      file.read((char*) &temp, sizeof(temp));
      arr[i] = temp;
    }
  }
  return arr;
}



std::unique_ptr<float[]> readMNISTData(int numberOfImages,
                                       int dataOfAnImage,
                                       const char *fname) {
  auto p = std::unique_ptr<float[]>(new float[numberOfImages * dataOfAnImage]);
  float *arr = &p[0];
  std::ifstream file(fname, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*) &number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char*) &n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char*) &n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    std::vector<unsigned char> buf(n_rows * n_cols);
    for(int i = 0; i < number_of_images; ++i) {
      file.read((char*) buf.data(), n_rows * n_cols);
      for(int r = 0; r < n_rows; ++r) {
        for(int c = 0; c < n_cols; ++c) {
          float pixel = ((float) buf[r * n_cols + c]) / 256;
          arr[i * dataOfAnImage + (n_rows * r) + c]= pixel;
        }
      }
    }
  }
  return p;
}
