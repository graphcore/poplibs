#include <poplib_test/Pass.hpp>

#include <iostream>
#include <popstd/exceptions.hpp>
#include <util/Compiler.hpp>

const char *poplib_test::asString(const Pass &pass) {
  switch (pass) {
  case Pass::ALL: return "all";
  case Pass::FWD: return "fwd";
  case Pass::BWD: return "bwd";
  case Pass::WU:  return "wu";
  }
  POPLIB_UNREACHABLE();
}

std::istream &poplib_test::operator>>(std::istream &is, Pass &pass) {
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
    throw popstd::poplib_error("Invalid pass <" + token + ">");
  return is;
}

std::ostream &poplib_test::operator<<(std::ostream &os, const Pass &pass) {
  return os << asString(pass);
}
