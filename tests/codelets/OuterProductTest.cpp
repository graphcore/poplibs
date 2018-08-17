// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the OuterProduct vertex
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poputil/TileMapping.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/Util.hpp>

#define BOOST_TEST_MODULE OuterProductTest
#include <boost/test/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplin;

//Define a number of tests to run:
struct TestParams  {
    unsigned insize;
    unsigned weightsize;
    unsigned matrices;
};

std::vector<TestParams> TestList={
    {4,12,2},
    {4,16,2},
    {5,18,3},
    {5,4,2},
    {1,4,1},
    {1,3,1},
    {8,12,1},
    {5,7,3},
    {4,1,2},
    {10,32,1}
};
//*************************************************
// Main Test function for OuterProduct vertex
//
// Overview:
// Define a test data array of size max_matrices * max_insize * max_weightsize
// Define an input array of size max_insize
// Define a weights array of size max_weightsize * max_matrices
//
// Fill the input,weights arrays with a test pattern of simple counting numbers
// inputs are non zero and even. weights non zero and odd.
//
// Each test will use a varying amount of the input and weights arrays,
// using a slice.  For tests where the output is smaller than its maximum, a
// slice into the output matrix array means that (as the
// output is initially zeroed) the output is tested for overwrites outside
// the intended area.
//*************************************************
void OuterProductTest(const Type &dataType) {

    //determine the sizes of arrays required
    auto test_count = TestList.size();

    auto max_insize = std::max_element(TestList.begin(),TestList.end(),
                [](TestParams &a, TestParams &b) {
                    return (a.insize < b.insize );})->insize;
    auto max_weightsize = std::max_element(TestList.begin(),TestList.end(),
                [](TestParams &a,TestParams &b) {
                    return (a.weightsize < b.weightsize );})->weightsize;
    auto max_matrices = std::max_element(TestList.begin(),TestList.end(),
                [](TestParams &a,TestParams &b) {
                    return (a.matrices < b.matrices );})->matrices;
    // Whole data array size
    auto total_size = max_insize * max_weightsize * max_matrices;

     // Program generated test data
    std::vector<double> outTest(total_size);

    //Input is the same for each matrix processed but weights is different
    std::vector<double> inTest(max_insize);
    std::vector<double> weightTest(max_weightsize * max_matrices);

    // Initialise input pattern.
    for (unsigned  i = 0; i < max_insize; i++)
            inTest[i] = 2*(i+1);
   // Initialise weight (input) pattern.
    for (unsigned  i = 0; i < max_weightsize * max_matrices; i++)
            weightTest[i] = 2*i +1;

    Device device = createTestDevice(TEST_TARGET);
    Target target=device.getTarget();

    //Create Graph object
    Graph graph(target);
    popops::addCodelets(graph);
    poplin::addCodelets(graph);

    //Input data
    Tensor in = graph.addVariable(dataType,{max_insize},
        "Input Data");
    graph.setTileMapping(in,0);

    Tensor weights = graph.addVariable(dataType,{max_matrices * max_weightsize},
        "Weight Data");
    graph.setTileMapping(weights,0);

    //Result data
    Tensor out = graph.addVariable(dataType,{max_matrices,
        max_insize * max_weightsize},"Output");
    graph.setTileMapping(out,0);

    //allocateHostMemoryForTensor
    Sequence uploadProg, downloadProg;
    std::vector<std::pair<std::string, char*>> tmap;
    auto input_host = allocateHostMemoryForTensor(in,"in",graph,uploadProg,
                                                  downloadProg,tmap);
    auto weight_host = allocateHostMemoryForTensor(weights,"weights",
                                                   graph,uploadProg,
                                                   downloadProg,tmap);

    auto output_host = allocateHostMemoryForTensor(out,"out",graph,uploadProg,
                                                   downloadProg,tmap);

    //Make multiple programs to test Transpose 2D each using
    //different input slices
    std::vector<Program> programs(test_count);



    for(int tests = 0; tests < test_count; tests++) {
        auto matrices = TestList[tests].matrices;
        auto insize = TestList[tests].insize;
        auto weightsize = TestList[tests].weightsize;

        Sequence sequence;

        ComputeSet testComputeSet = graph.addComputeSet("computeOuterProduct");

        const auto vertexClass = templateVertex("poplin::OuterProduct",
                dataType);

        auto Vertex = graph.addVertex(testComputeSet,vertexClass);
        graph.setTileMapping(Vertex,0);

        //Different slices of the same input data to test looping decisions
        auto sliceIn = in.slice(0,insize);
        auto sliceWeights = weights.slice(0,matrices * weightsize);
        auto sliceOut = out.slice({0,0},{matrices,insize * weightsize});

        graph.connect(Vertex["in"],sliceIn);
        graph.connect(Vertex["weights"],sliceWeights);
        graph.connect(Vertex["out"],sliceOut);

        graph.setInitialValue(Vertex["chansPerGroup"], weightsize);

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
        auto matrices = TestList[tests].matrices;
        auto insize = TestList[tests].insize;
        auto weightsize = TestList[tests].weightsize;

        copy(target,inTest.data(),inTest.size(),dataType,input_host.get());
        copy(target,weightTest.data(),weightTest.size(),dataType,
                weight_host.get());

        engine.run(uploadProgIndex);

        engine.run(tests);

        engine.run(downloadProgIndex);
        copy(target,dataType,output_host.get(),outHost.data(),outHost.size());

        // Host generated result, start with zeros
        for(unsigned i = 0; i < total_size; i++)
            outTest[i] = 0;

        for(int i = 0; i < matrices; i++){
            for(int j = 0; j < insize; j++){
                for(int k = 0; k<weightsize; k++){
                    outTest[k + j *weightsize + i *max_weightsize *max_insize] =
                        inTest[j]*weightTest[k + i *(weightsize)];
                }
            }
        }
         // Check the result, in the outTest array
        // Always check the whole output memory to catch any overwrites
        bool check = checkIsClose("Test_"+std::to_string(tests),
            outHost.data(),{outHost.size()},outTest.data(),outTest.size(),
            0.0,0.0);
        BOOST_CHECK(check);
    }

}
 BOOST_AUTO_TEST_CASE(OuterProductTest_float) {OuterProductTest(FLOAT);}
 BOOST_AUTO_TEST_CASE(OuterProductTest_half) {OuterProductTest(HALF);}
