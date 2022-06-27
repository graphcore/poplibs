# Note that although this is purely introspection-based,
# we need an IpuModel in order to have > 1 tile.
add_unit_test(IdenticalLayoutTest IdenticalLayoutTest.cpp
              VARIANTS IpuModel2)
