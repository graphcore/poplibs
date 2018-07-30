const std::vector<std::vector<bool>> in1 = {
  {},
  {true},
  {true,  true},
  {false, false, false},
  {false, false, false, false},
  {true, false, true, false, true},
  {0, 1, 0, 1, 0, 1, 0, 1, 0}
};

const std::vector<std::vector<bool>> in2 = {
  {},
  {false},
  {false, false},
  {true,  true,  true},
  {true,  true,  true, true},
  {false, true, false, true, false},
  {1, 0, 1, 0, 1, 0, 1, 0, 1}
};

const std::vector<std::vector<bool>> expected = {
  {},
  { true},
  { false, true},
  { true,  false, true},
  { false, true,  false, true},
  { true,  false, false, true, true},
  { 1, 0, 1, 1, 0, 1, 1, 1, 1}
};
