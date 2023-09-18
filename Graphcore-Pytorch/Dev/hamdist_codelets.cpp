// This code is based on Poplar Tutorial 6: Matrix-vector Multiplication Optimisation example from Graphcore, 
// and modified to performs a hamming distance operation between a 2D matrix of RxC with a vector size of C.

// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>

using namespace poplar;

// This file contains the definitions of the vertex types used by the
// matrix-vector hamming distance example. The vertex type provides a description
// of how individual tasks on the IPU will perform computation.

// A vertex type to perform a dot product calculation.
class HamDistanceVertex : public Vertex {
public:
  // These two inputs read a vector of values (that is, an ordered, contiguous
  // set of values in memory) from the graph.
  Input<Vector<int>> a;
  Input<Vector<int>> b;

  // The output is to a single scalar value in the graph.
  Output<int> out;

  // The compute method performs the hamming distance between inputs 'a' and
  // 'b' and stores the result in 'out'.
  bool compute() {
    int res = 0;
    for (unsigned i = 0; i < a.size(); ++i)
      res += a[i] ^ b[i];
    *out = __builtin_popcount(res); // count set bits after xor
    return true;
  }
};

// A vertex type to sum up a set of inputs.
class ReduceVertex : public Vertex {
public:
  Input<Vector<int>> in;
  Output<int> out;

  bool compute() {
    int res = 0;
    for (unsigned i = 0; i < in.size(); ++i) {
      res += in[i];
    }
    *out = res;
    return true;
  }
};