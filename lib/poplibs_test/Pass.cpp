// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <poplibs_test/Pass.hpp>

#include <iostream>
#include <poplibs_support/Compiler.hpp>
#include <poputil/exceptions.hpp>

const char *poplibs_test::asString(const Pass &pass) {
  switch (pass) {
  case Pass::ALL:
    return "all";
  case Pass::FWD:
    return "fwd";
  case Pass::BWD:
    return "bwd";
  case Pass::WU:
    return "wu";
  }
  POPLIB_UNREACHABLE();
}

std::istream &poplibs_test::operator>>(std::istream &is, Pass &pass) {
  std::string token;
  is >> token;
  if (token == "all")
    pass = Pass::ALL;
  else if (token == "fwd")
    pass = Pass::FWD;
  else if (token == "bwd")
    pass = Pass::BWD;
  else if (token == "wu")
    pass = Pass::WU;
  else
    throw poputil::poplibs_error("Invalid pass <" + token + ">");
  return is;
}

std::ostream &poplibs_test::operator<<(std::ostream &os, const Pass &pass) {
  return os << asString(pass);
}
