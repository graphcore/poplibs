// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the Cast vertex
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/Util.hpp>

#define BOOST_TEST_MODULE CastTest
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;


//Define a number of tests to run:
struct TestParams {
    unsigned columns;
    unsigned offsetOut;
};

// Tests - based on knowing that the implementation
// Has execution paths for multiples of 4, and needs to tidy the last
// few following it.  Also a long test to check that sizes > maximum
// repeat count function correctly. The output can either be 8 byte or 4 byte
// aligned. offsetOut of 2 equates to 4 byte aligned only.
std::vector<TestParams> TestList={
    {1,0},      {1,2},
    {2,0},      {2,2},
    {3,0},      {3,2},
    {4,0},      {4,2},
    {16,0},     {16,2},
    {17,0},     {17,2},
    {18,0},     {18,2},
    {19,0},     {19,2},
    {0x1100,0}, {0x1100,2},
};

//*************************************************
// Main Test function for Cast
//
// Overview:
//
// Run a series of tests that cast a varying number of items.
// The results are put into a memory area large enough to
// hold the largest test result, so often the other items are
// expected to be zero.  This is checked as well as the "wanted" data.
//*************************************************
void CastTest(const Type &dataTypeIn, const Type &dataTypeOut) {

    //determine the sizes of arrays required
    auto test_count=TestList.size();

    auto max_cols=std::max_element(TestList.begin(),TestList.end(),
                [](TestParams &a, TestParams &b) {
                    return (a.columns <b.columns );})->columns;
    auto max_offsetOut=std::max_element(TestList.begin(),TestList.end(),
                [](TestParams &a, TestParams &b) {
                    return (a.offsetOut <b.offsetOut );})->offsetOut;
    //Whole data array size
    auto total_size=max_cols + max_offsetOut;

    // Program generated test data
    std::vector<double> outTest(total_size);
    std::vector<double> inTest(max_cols);

    // Initialise input pattern, picking a numeric range and
    // tolerance (below) that works for halves as a limited size/resolution data
    // type with enough unique numbers to satisfy a large test size
    for (unsigned  i = 0; i < max_cols; i++)
            inTest[i] = 0.1 * i + 1;

    Device device = createTestDevice(TEST_TARGET);
    Target target=device.getTarget();

    //Create Graph object
    Graph graph(target);
    popops::addCodelets(graph);

    //Input data
    Tensor in=graph.addVariable(dataTypeIn,{max_cols}, "Input Data");
    graph.setTileMapping(in,0);

    //Result data
    Tensor out=graph.addVariable(dataTypeOut,{total_size}, "Output");
    graph.setTileMapping(out,0);

    //allocateHostMemoryForTensor
    Sequence uploadProg, downloadProg;
    std::vector<std::pair<std::string, char*>> tmap;
    auto input=allocateHostMemoryForTensor(in,"in",graph,uploadProg,
                                                  downloadProg,tmap);

    auto output=allocateHostMemoryForTensor(out,"out",graph,uploadProg,
                                                  downloadProg,tmap);

    //Make multiple programs to test Cast each using
    //different input slices, for different input sizes and offsets
    std::vector<Program> programs(test_count);

    for(int tests=0;tests<test_count;tests++) {
        auto columns=TestList[tests].columns;
        auto offsetOut=TestList[tests].offsetOut;

        Sequence sequence;

        ComputeSet testComputeSet=graph.addComputeSet("computeCast");

        const auto vertexClass=templateVertex("popops::Cast",dataTypeIn,
            dataTypeOut);

        auto castVertex=graph.addVertex(testComputeSet,vertexClass);
        graph.setTileMapping(castVertex,0);

        //Different slices of the same input data to test looping decisions,
        // Slices of the output data with offset
        auto sliceIn=in.slice(0,columns);
        auto sliceOut=out.slice(offsetOut, columns+ offsetOut);

        graph.connect(castVertex["src"],sliceIn);
        graph.connect(castVertex["dst"],sliceOut);

        popops::zero(graph,out,sequence,"Zero output");
        sequence.add(Execute(testComputeSet));
        programs[tests]=sequence;
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

    for(int tests=0;tests<test_count;tests++) {
        auto columns=TestList[tests].columns;
        auto offsetOut=TestList[tests].offsetOut;

        copy(target,inTest.data(),inTest.size(),dataTypeIn,input.get());

        engine.run(uploadProgIndex);

        engine.run(tests);

        engine.run(downloadProgIndex);

        copy(target,dataTypeOut,output.get(),outHost.data(),outHost.size());

        //Host generated result, start with zeros
         for(unsigned i=0;i<total_size;i++)
            outTest[i]=0;
        //Then cast the same portion of the input as the code under test
        for(unsigned j=0; j<columns; j++) {
            outTest[j + offsetOut]=inTest[j];
        }

        //Check the result, in the outTest array
        //Always check the whole output memory to catch any overwrites
        //
        bool check=checkIsClose("Test_"+std::to_string(tests),
            outHost.data(),{outHost.size()},outTest.data(),outTest.size(),
            0.05,0.05);
        BOOST_CHECK(check);
    }
}
 BOOST_AUTO_TEST_CASE(CastTest_float_half) {CastTest(FLOAT,HALF);}
