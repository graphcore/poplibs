// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef popops_codelets_util_hpp
#define popops_codelets_util_hpp

template <typename T> const T &max(const T &x, const T &y) {
  return x < y ? y : x;
}

template <typename T> const T &min(const T &x, const T &y) {
  return x < y ? x : y;
}

#endif // popops_codelets_util_hpp
