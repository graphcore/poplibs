// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE ConvOptionsTest
#include "ConvOptions.hpp"
#include <boost/test/unit_test.hpp>

#include <boost/property_tree/ptree.hpp>

using namespace poplin;
using namespace poplin::internal;
using namespace boost::property_tree;

BOOST_AUTO_TEST_CASE(ValidatePartitionVars) {
  ptree t;
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartitionVars("", t));
  t.push_back(ptree::value_type("0", ptree("4")));
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartitionVars("", t));
  t.push_back(ptree::value_type("2", ptree("5")));
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartitionVars("", t));
  t.push_back(ptree::value_type("5", ptree("a watched pot never boils")));
  BOOST_CHECK_THROW(validatePlanConstraintsPartitionVars("", t),
                    poplar::invalid_option);
  t = ptree("0");
  BOOST_CHECK_THROW(validatePlanConstraintsPartitionVars("", t),
                    poplar::invalid_option);
  t.clear();
  t.put<std::string>("hello", "world");
  BOOST_CHECK_THROW(validatePlanConstraintsPartitionVars("", t),
                    poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ValidatePartitionSplitVar) {
  ptree t;
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartitionSplitVar("", t));
  t.push_back(ptree::value_type("serial", ptree("4")));
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartitionSplitVar("", t));
  t.push_back(ptree::value_type("parallel", ptree("5")));
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartitionSplitVar("", t));
  t.clear();
  t.push_back(ptree::value_type("serial", ptree("a watched pot never boils")));
  BOOST_CHECK_THROW(validatePlanConstraintsPartitionSplitVar("", t),
                    poplar::invalid_option);
  t = ptree("0");
  BOOST_CHECK_THROW(validatePlanConstraintsPartitionSplitVar("", t),
                    poplar::invalid_option);
  t.clear();
  t.put<std::string>("hello", "world");
  BOOST_CHECK_THROW(validatePlanConstraintsPartitionSplitVar("", t),
                    poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ValidateTransform) {
  ptree t;
  BOOST_CHECK_NO_THROW(validatePlanConstraintsTransform("", t));
  t.push_back(ptree::value_type("swapOperands", ptree("false")));
  BOOST_CHECK_NO_THROW(validatePlanConstraintsTransform("", t));
  t.push_back(ptree::value_type("a watched pot never boils", {}));
  BOOST_CHECK_THROW(validatePlanConstraintsTransform("", t),
                    poplar::invalid_option);
  t.clear();
  t.push_back(
      ptree::value_type("swapOperands", ptree("a watched pot never boils")));
  BOOST_CHECK_THROW(validatePlanConstraintsTransform("", t),
                    poplar::invalid_option);
  t = ptree("0");
  BOOST_CHECK_THROW(validatePlanConstraintsTransform("", t),
                    poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ValidatePartition) {
  ptree t;
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartition("", t));
  t.push_back(ptree::value_type("convGroupSplit", ptree("4")));
  BOOST_CHECK_NO_THROW(validatePlanConstraintsPartition("", t));
  t.push_back(ptree::value_type("a watched pot never boils", {}));
  BOOST_CHECK_THROW(validatePlanConstraintsPartition("", t),
                    poplar::invalid_option);
  t.clear();
  t.push_back(
      ptree::value_type("convGroupSplit", ptree("a watched pot never boils")));
  BOOST_CHECK_THROW(validatePlanConstraintsPartition("", t),
                    poplar::invalid_option);
  t = ptree("0");
  BOOST_CHECK_THROW(validatePlanConstraintsPartition("", t),
                    poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ValidateTopLevel) {
  ptree t;
  ValidateConvPlanConstraintsOption uut;

  BOOST_CHECK_NO_THROW(uut(t));
  t.push_back(ptree::value_type("0", {}));
  BOOST_CHECK_NO_THROW(uut(t));
  t.push_back(ptree::value_type("2", {}));
  BOOST_CHECK_NO_THROW(uut(t));
  t.push_back(ptree::value_type(" 5", {}));
  BOOST_CHECK_NO_THROW(uut(t));
  t.push_back(ptree::value_type("", {}));
  BOOST_CHECK_THROW(uut(t), poplar::invalid_option);
  t.push_back(ptree::value_type("-1", {}));
  BOOST_CHECK_THROW(uut(t), poplar::invalid_option);
  t.clear();
  t.push_back(ptree::value_type("a watched pot never boils", {}));
  BOOST_CHECK_THROW(uut(t), poplar::invalid_option);
  t = ptree("0");
  BOOST_CHECK_THROW(uut(t), poplar::invalid_option);
}
