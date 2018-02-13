#pragma once

template <typename T>
static const T &max(const T &x, const T &y) {
  return x < y ? y : x;
}

template <typename T>
static const T &min(const T &x, const T &y) {
  return x < y ? x : y;
}
