#define BOOST_TEST_MODULE PlanConstraintsTest
#include <boost/test/unit_test.hpp>
#include "poplibs_support/PlanConstraints.hpp"

#include <boost/property_tree/ptree.hpp>

using namespace poplibs_support;
using namespace boost::property_tree;

BOOST_AUTO_TEST_CASE(ValidateBoolean) {
  ptree t;
  BOOST_CHECK_THROW(validatePlanConstraintsBoolean("", t),
                    poplar::invalid_option);
  t = ptree("true");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsBoolean("", t));
  t = ptree("false");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsBoolean("", t));
  t = ptree("0");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsBoolean("", t));
  t = ptree("1");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsBoolean("", t));
  t = ptree("");
  BOOST_CHECK_THROW(validatePlanConstraintsBoolean("", t),
                    poplar::invalid_option);
  t = ptree("2");
  BOOST_CHECK_THROW(validatePlanConstraintsBoolean("", t),
                    poplar::invalid_option);
  t = ptree("-1");
  BOOST_CHECK_THROW(validatePlanConstraintsBoolean("", t),
                    poplar::invalid_option);
  t = ptree("a watched pot never boils");
  BOOST_CHECK_THROW(validatePlanConstraintsBoolean("", t),
                    poplar::invalid_option);
  t.clear();
  t.put<std::string>("key", "value");
  BOOST_CHECK_THROW(validatePlanConstraintsBoolean("", t),
                    poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ValidateUnsigned) {
  ptree t;
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
  t = ptree("true");
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
  t = ptree("false");
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
  t = ptree("0");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsUnsigned("", t));
  t = ptree("1");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsUnsigned("", t));
  t = ptree("2");
  BOOST_CHECK_NO_THROW(validatePlanConstraintsUnsigned("", t));
  t = ptree("");
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
  t = ptree("-1");
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
  t = ptree("a watched pot never boils");
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
  t.clear();
  t.put<std::string>("key", "value");
  BOOST_CHECK_THROW(validatePlanConstraintsUnsigned("", t),
                    poplar::invalid_option);
}
