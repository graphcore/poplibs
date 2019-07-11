#define BOOST_TEST_MODULE ConvPlanTest
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/test/unit_test.hpp>
#include <poplin/internal/ConvPlan.hpp>
#include "poplin/internal/ConvOptions.hpp"
#include <popnn/codelets.hpp>
#include <vector>
#include "poputil/exceptions.hpp"
#include "poplin/CanonicalConvParams.hpp"

static auto params = poplin::ConvParams{poplar::FLOAT, // Data type
                                        1,             // batch size
                                        {4, 4},        // input field shape
                                        {3, 3},        // kernel shape
                                        1,             // input channels
                                        1,             // output channels
                                        1};            // conv groups

BOOST_AUTO_TEST_CASE(getPlan){
  poplar::Graph graph(poplar::Target::createCPUTarget());
  auto &target = graph.getTarget();
  auto options = poplin::ConvOptions(target.getNumIPUs(),
                                     target.getTilesPerIPU());

  poplin::getPlan(target, params, options, nullptr);
}

BOOST_AUTO_TEST_CASE(getCachedPlans) {
  poplar::Graph graph(poplar::Target::createIPUTarget(2, "ipu0"));
  auto &target = graph.getTarget();

  poplin::PlanningCache cache;

  poplin::getPlan(target, params,
                  poplin::ConvOptions(target.getNumIPUs(),
                                      target.getTilesPerIPU()),
                  &cache);
  poplin::getPlan(target, params,
                  poplin::ConvOptions(1u,
                                      target.getTilesPerIPU() / 2),
                  &cache);
}

BOOST_AUTO_TEST_CASE(VirtualGraphIPUCheck){
  poplar::Graph graph(poplar::Target::createIPUTarget(2, "ipu0"));
  auto &target = graph.getTarget();

  poplin::PlanningCache cache;

  BOOST_CHECK_THROW(
      poplin::getPlan(target, params,
                      poplin::ConvOptions(target.getNumIPUs(),
                                          target.getTilesPerIPU() * 2),
                      &cache),
                      poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(VirtualGraphTilesCheck){
  poplar::Graph graph(poplar::Target::createIPUTarget(2, "ipu0"));
  auto &target = graph.getTarget();

  poplin::PlanningCache cache;

  BOOST_CHECK_THROW(
      poplin::getPlan(target, params,
                      poplin::ConvOptions(target.getNumIPUs() + 1,
                                          target.getTilesPerIPU()),
                      &cache),
                      poputil::poplibs_error);
}


// Test some simple aspects of plan constraining that we currently support
// TODO: More comprehensive testing of failure cases. This just checks
// that some existing options do what was expected and doesn't check for
// expected failure modes (accidentally specifying something that isn't
// handled for instance).
BOOST_AUTO_TEST_CASE(ConstrainPlan) {
  poplar::Graph graph(poplar::Target::createIPUTarget(1, "ipu1"));
  auto &target = graph.getTarget();

  poplin::PlanningCache cache;

  using namespace boost::property_tree;
  std::stringstream ss;
  ss << R"delim(
    {"0": {"transform": {"swapOperands": true}}}
  )delim";
  ptree t;
  json_parser::read_json(ss, t);
  auto options = poplin::ConvOptions(target.getNumIPUs(),
                                     target.getTilesPerIPU());
  options.planConstraints = std::move(t);
  auto plan = poplin::getPlan(target, params, options, &cache);
  BOOST_CHECK_EQUAL(plan.transforms[0].swapOperands, true);
  ss.str(""); ss.clear();
  ss << R"delim(
    {"0": {"partition": {"fieldSplit": {"0": 2, "1": 2}}}}
  )delim";
  t.clear();
  json_parser::read_json(ss, t);
  options.planConstraints = std::move(t);
  plan = poplin::getPlan(target, params, options, &cache);
  BOOST_CHECK_EQUAL(plan.partitions[0].fieldSplit[0], 2);
  BOOST_CHECK_EQUAL(plan.partitions[0].fieldSplit[1], 2);
}
