// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
const std::vector<std::vector<int>> in1 = {{},
                                           {11},
                                           {31, 33},
                                           {51, 53, 55},
                                           {71, 73, 75, 77},
                                           {91, 93, 95, 97, 99},
                                           {1, 3, 5, 7, 9, 21, 23, 25, 27}};

const std::vector<std::vector<int>> in2 = {{},
                                           {20},
                                           {40, 42},
                                           {60, 62, 64},
                                           {80, 82, 84, 86},
                                           {100, 102, 104, 106, 108},
                                           {2, 4, 6, 8, 10, 22, 24, 26, 28}};

const std::vector<std::vector<int>> expected = {
    {},
    {11},
    {40, 33},
    {60, 53, 64},
    {71, 82, 75, 86},
    {91, 93, 104, 106, 99},
    {2, 4, 6, 7, 9, 21, 24, 25, 28}};
