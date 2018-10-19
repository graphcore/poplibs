// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the Dynamic Slice adn Dynamic Slice update 2d vertices
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/Util.hpp>

#define BOOST_TEST_MODULE DynamicSliceCodeletTest
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;


//Define a number of tests to run:
struct TestParams {
  unsigned offset;
  unsigned numBaseElements;
  unsigned numSubElements;
  unsigned columns;
  unsigned dstOffset;
  bool     update;
};

std::vector<TestParams> TestList={
   {0, 1, 2, 1, 1, false},
   {0, 1, 2, 2, 1, false},
   {0, 1, 2, 3, 1, false},
   {0, 1, 2, 4, 1, false},
   {0, 1, 2, 5, 1, false},
   {0, 1, 2, 6, 1, false},
   {1, 3, 2, 7, 0, false},
   {0, 4, 4, 8, 1, false},
   {2, 4, 5, 9, 0, false},
   {0, 4, 4, 10, 1, false},
   {2, 4, 5, 11, 0, false},
   {0, 2, 2, 12, 1, false},
   {3, 5, 5, 13, 0, false},
   {3, 5, 5, 31, 0, false},

   {0, 1, 1, 6, 1, true},
   {1, 2, 2, 7, 0, true},
   {0, 4, 4, 8, 1, true},
   {2, 4, 4, 9, 0, true},
   {0, 2, 2, 12, 1, true},
   {3, 5, 5, 13, 0, true},

};
//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void DynamicSliceSupervisorHost ( unsigned offset,
  std::vector<double> &baseT,
  std::vector<double> &subT,
  unsigned short numBaseElements,
  unsigned short numSubElements,
  unsigned short regionSize,
  unsigned short dstOffset)
{
  unsigned baseSlice = offset;

  if (baseSlice >= numBaseElements)
    baseSlice=0;
  for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
    for (unsigned e = 0; e != regionSize; e++) {
      subT[subSlice * regionSize + e + dstOffset] =
        baseT[baseSlice * regionSize + e];
    }
    baseSlice++;
    if (baseSlice >= numBaseElements)
      baseSlice=0;
  }
}
//*************************************************
// C test function, based on the original C version of the vertex
//*************************************************
void DynamicUpdateSliceSupervisorHost ( unsigned offset,
  std::vector<double> &baseT,
  std::vector<double> &subT,
  unsigned short numBaseElements,
  unsigned short numSubElements,
  unsigned short regionSize,
  unsigned short dstOffset)
{
  unsigned baseSlice = offset;

  if (baseSlice >= numBaseElements)
    baseSlice=0;
  for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
    for (unsigned e = 0; e != regionSize; e++) {
      baseT[baseSlice * regionSize + e + dstOffset] =
        subT[subSlice * regionSize + e];
    }
    baseSlice++;
    if (baseSlice >= numBaseElements)
      baseSlice=0;
  }
}

//*************************************************
// Main Test function for DynamicSliceSupervisor, DynamicUpdateSliceSupervisor
//
// Overview:
//
// Output memory space is initialised as all zero.
// Input memory space is intitalised with a simple test pattern
// Run a series of tests that copy a varying number of items.
// The results are put into a memory area large enough to
// hold the largest test result, so often the other items are
// expected to be zero.  This is checked as well as the "wanted" data.
//*************************************************
void DynamicSliceCodeletTest(const Type &dataType) {

  //determine the sizes of arrays required
  auto test_count=TestList.size();

  const auto maxColumns = std::max_element(TestList.begin(),TestList.end(),
              [](TestParams &a, TestParams &b) {
                  return (a.columns < b.columns);})->columns;

  const auto maxDstOffset = std::max_element(TestList.begin(),TestList.end(),
              [](TestParams &a, TestParams &b) {
              return (a.dstOffset < b.dstOffset);})->dstOffset;

  // Check max sizes of regions so that the test method generates legal copies
  const auto maxElements = std::max_element(TestList.begin(),TestList.end(),
              [](TestParams &a, TestParams &b) {
              return (std::max(a.numSubElements, a.numBaseElements) <
                      std::max(b.numSubElements, b.numBaseElements));});

  const auto maxRows= std::max( maxElements->numBaseElements,
                      maxElements->numSubElements);
  // Whole data array size - oversize foe the smaller tests
  // so we verify areas not overwritten
  auto total_size = maxColumns * maxRows + maxDstOffset;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_size);

  // Initialise input pattern, dummy data to check its overwritten when
  // it should be, and not when its not
  for (unsigned  i = 0; i < total_size; i++)
          inTest[i] = i + 1;

  Device device = createTestDevice(TEST_TARGET);
  Target target=device.getTarget();

  //Create Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  // Test In and out tensor
  Tensor in=graph.addVariable(dataType,{total_size},"Input");
  Tensor out=graph.addVariable(dataType,{total_size},"Output");
  graph.setTileMapping(in,0);
  graph.setTileMapping(out,0);

  //allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto input=allocateHostMemoryForTensor(in,"in",graph,uploadProg,
                                                downloadProg,tmap);
  auto output=allocateHostMemoryForTensor(out,"out",graph,uploadProg,
                                                downloadProg,tmap);

  //Make multiple programs to test dynamic slice, each selecting
  //different slices, for different output sizes and offsets
  std::vector<Program> programs;

  for(unsigned tests = 0; tests < test_count; tests++) {
    auto offset=TestList[tests].offset;
    auto numBaseElements=TestList[tests].numBaseElements;
    auto numSubElements=TestList[tests].numSubElements;
    auto columns=TestList[tests].columns;
    auto dstOffset=TestList[tests].dstOffset;
    auto update=TestList[tests].update;

    Sequence sequence;

    ComputeSet testComputeSet=graph.addComputeSet("computeDynamicSlice");

    auto vertexClass = templateVertex("popops::DynamicSliceSupervisor",
          dataType);
    auto base=in.slice(0, numBaseElements * columns);
    auto sub=out.slice(dstOffset, numSubElements * columns + dstOffset);
    if(update) {
      vertexClass=templateVertex("popops::DynamicUpdateSliceSupervisor",
        dataType);
      base=out.slice(dstOffset, numBaseElements * columns + dstOffset);
      sub=in.slice(0, numSubElements * columns);
    }

    auto dsVertex=graph.addVertex(testComputeSet,
                                            vertexClass,
                                            {{"offset", offset},
                                            {"baseT", base},
                                            {"subT", sub}
                                            });
    graph.setInitialValue(dsVertex["numBaseElements"], numBaseElements);
    graph.setInitialValue(dsVertex["numSubElements"], numSubElements);
    graph.setInitialValue(dsVertex["regionSize"], columns);
    graph.setInitialValue(dsVertex["numWorkers"],target.getNumWorkerContexts());
    graph.setTileMapping(dsVertex,0);

    popops::zero(graph,out, sequence,"Zero output");
    sequence.add(Execute(testComputeSet));
    programs.push_back(sequence);

  }

  const auto uploadProgIndex = programs.size();
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size();
  programs.push_back(std::move(downloadProg));

  //Run each program and compare host and IPU result
  Engine engine(graph,programs);
  engine.load(device);
  attachStreams(engine, tmap);

  //Put test inputs into an array of the correct type ready to use
  std::vector<double> outHost(total_size);

  for(unsigned tests = 0; tests < test_count; tests++) {
    auto offset=TestList[tests].offset;
    auto numBaseElements=TestList[tests].numBaseElements;
    auto numSubElements=TestList[tests].numSubElements;
    auto columns=TestList[tests].columns;
    auto dstOffset=TestList[tests].dstOffset;
    auto update=TestList[tests].update;

    copy(target,inTest.data(),inTest.size(),dataType,input.get());


    engine.run(uploadProgIndex);
    engine.run(tests);
    engine.run(downloadProgIndex);

    copy(target,dataType,output.get(),outHost.data(),outHost.size());

    //Host generated result, start with 0s
     for(unsigned i=0;i<total_size;i++)
        outTest[i]=0;

    // Run the host version of the codelet to compare against - either
    // update or non update version
    if(update) {
      DynamicUpdateSliceSupervisorHost(offset,
                        outTest,
                        inTest,
                        numBaseElements,
                        numSubElements,
                        columns,
                        dstOffset);
    }
    else {
      DynamicSliceSupervisorHost(offset,
                        inTest,
                        outTest,
                        numBaseElements,
                        numSubElements,
                        columns,
                        dstOffset);
    }

    //Check the result, in the outTest array
    //Always check the whole output memory to catch any overwrites
    bool check=checkIsClose("Test_"+std::to_string(tests),
      outHost.data(),{outHost.size()},outTest.data(),outTest.size(),
      0.0,0.0);
    BOOST_CHECK(check);
  }
}
  BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_float) {
                DynamicSliceCodeletTest(FLOAT);}
  BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_half) {
                DynamicSliceCodeletTest(HALF);}
  BOOST_AUTO_TEST_CASE(DynamicSliceSupervisorCodeletTest_int) {
                DynamicSliceCodeletTest(INT);}
