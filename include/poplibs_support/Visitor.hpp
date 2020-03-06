// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#ifndef poplibs_include_core_util_Visitor_hpp
#define poplibs_include_core_util_Visitor_hpp

#include <boost/variant/static_visitor.hpp>

namespace poplibs_support {

// There is a much simpler way of doing this with C++17 (see cppreference's
// std::visit() example).

// Idea from https://stackoverflow.com/a/7870614/265521

// This uses variadic templates to construct an inheritance chain of lambda
// functions. A lambda function [](int) { foo(); } is essentially equivalent to
//
// struct {
//   void operator()(int) { foo(); }
// };
//
// We want a visitor to be like:
//
// struct Visitor : boost::static_visitor<void> {
//   void operator()(int) { foo(); }
//   void operator()(string) { bar(); }
//   ...
// };
//
// Which we can do with an inheritance chain like this:
//
// struct Visitor1 : boost::static_visitor<void>, Lambda1 {
//   using Lambda1::operator();
// };
//
// struct Visitor2 : Visitor1, Lambda2 {
//   using Visitor1::operator();
//   using Lambda2::operator();
// };
//
// ...
//
// struct VisitorN : VisitorN-1, LambdaN {
//   using VisitorN-1::operator();
//   using LambdaN::operator();
// };
//
// And so on, where `Lamdba1`, `Lambda2`, ... `LambdaN` are the lambda
// functions. This can  be done using variadic templates.

// The variadic template type.
template <typename ReturnType, typename... Lambdas> struct Visitor;

// A visitor that inherits from a single lambda. This is the terminal element
// of the variadic chain, equivalent to Visitor1 above.
template <typename ReturnType, typename Lambda>
struct Visitor<ReturnType, Lambda> : public boost::static_visitor<ReturnType>,
                                     public Lambda {

  explicit Visitor(Lambda l) : boost::static_visitor<ReturnType>(), Lambda(l) {}

  using Lambda::operator();
};

// A visitor constructed from more than one lambda. It creates a new
// class in the inheritance chain from the first lambda. This is equivalent
// to VisitorN above.
template <typename ReturnType, typename HeadLambda, typename... TailLambdas>
struct Visitor<ReturnType, HeadLambda, TailLambdas...>
    : public Visitor<ReturnType, TailLambdas...>, public HeadLambda {

  Visitor(HeadLambda head, TailLambdas... tail)
      : Visitor<ReturnType, TailLambdas...>(tail...), HeadLambda(head) {}

  using Visitor<ReturnType, TailLambdas...>::operator();
  using HeadLambda::operator();
};

// This is to help template type deduction.
template <typename ReturnType, typename... Lambdas>
Visitor<ReturnType, Lambdas...> make_visitor(Lambdas... lambdas) {
  return Visitor<ReturnType, Lambdas...>(lambdas...);
}

} // namespace poplibs_support

#endif // poplar_include_core_util_Visitor_hpp
