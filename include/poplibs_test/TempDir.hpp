// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef test_helper_hpp
#define test_helper_hpp

#include <boost/filesystem.hpp>

/// Representation of temporary directory that removes it on destruction.
class TempDir {
  const std::string path;

public:
  TempDir(std::string path) : path{std::move(path)} {}
  static TempDir create() {
    using namespace boost::filesystem;
    const auto path = unique_path("poplibs_%%%%%%%%%%%%");
    if (!create_directories(path)) {
      throw std::runtime_error("Error creating temporary directory " +
                               path.string());
    }
    return TempDir(path.string());
  }
  TempDir(const TempDir &) = delete;
  TempDir &operator=(const TempDir &) = delete;
  TempDir(TempDir &&other) : path{std::move(other.path)} {}
  ~TempDir() {
    using namespace boost::filesystem;
    if (!path.empty()) {
      if (exists(path)) {
        remove_all(path);
      }
    }
  }
  std::string getPath() const { return path; }
};

#endif
