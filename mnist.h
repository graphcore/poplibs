#ifndef _MNIST_H_
#define _MNIST_H_
#include <vector>
#include <memory>

std::unique_ptr<unsigned[]> readMNISTLabels(int NumberOfImages,
                                            const char *fname);

std::unique_ptr<float[]> readMNISTData(int numberOfImages,
                                       int dataOfAnImage,
                                       const char *fname);

#endif //_MNIST_H_
