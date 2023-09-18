// This code is based on Poplar Tutorial 6: Matrix-vector Multiplication Optimisation example from Graphcore, 
// and modified to perform a hamming distance operation between a 2D matrix of RxC with a vector size of C.

// Compile:
// g++ --std=c++11 hamdist.cpp -lpoplar -lpoputil -o hamdist_test
// To run:
// ./hamdist_test 8192 256 --device ipu


// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <cstdlib>
#include <chrono>

// This example performs matrix hamming distance with a vector on the IPU by decomposing
// the column axis (that is, the number of columns) into N partial sums.
// The partial sums are then added up to get the final result. For example,
// consider the following multiplication:
//
// ( 0 1 2 3 )   ( A )
// ( 4 5 6 7 ) ^ ( B )
//               ( C )
//               ( D )
//
// With a column axis split of two there would be four
// partial sums:
//
//  P1 = popcount(0^A) + popcount(1^B), P2 = popcount(C^2) + popcount(D^3), 
//  P3 = popcount(4^A) + popcount(5^B), P4 = popcount(C^6) + popcount(D^7)
//
// The final vector would be created by adding these partial sums:
//
//  ( P1 + P2, P3 + P4)
//
// On the IPU, the code needs to work out the optimal column axis split for
// a particular matrix. This is done by estimating the compute cost for
// every possible split and choosing the best.

#define VERBOSE_LEVEL 1

using namespace poplar;
using namespace poplar::program;


// Utility function to divide two integers rounding up the result.
inline unsigned ceilDiv(unsigned a, unsigned b) { return (a + b - 1) / b; }

// Function to estimate the number of cycles required to perform the
// multiplication given a particular column axis split.
unsigned estimateCycles(const Graph &graph, unsigned numRows, unsigned numCols,
                        unsigned split, bool verbose) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  // There are numRows * split partial sums, each will have a vertex to
  // calculate it and these vertices will be spread evenly over the tiles.
  // The compute cost is equal to the time of the longest running tile (that is,
  // the tile with the maximum number of vertices).
  unsigned maxVerticesPerTile =
      std::max(ceilDiv(numRows * split, numTiles), 1U);
  // Each vertex has to calculate a dot product which it can do using
  // a vector unit of width 2 (that is, 2 macs per cycle).
  auto cyclesPerVertex = 5 + (ceilDiv(numCols, split) + 1) / 2;
  // The compute cost is the time of the longest running worker context on
  // the longest running tile. To calculate this we can round up the
  // number of vertices to the nearest multiple of the number of worker
  // contexts.
  auto computeCost =
      ceilDiv(maxVerticesPerTile, numWorkers) * numWorkers * cyclesPerVertex;
  // The number of exchange cycles before the partial sum calculation is
  // bound by the maximum number of bytes a tile can receive.
  // Each tile will receive multiple sections of the input vector
  // (of size 'numCols / split'). Each tile will be allocated all the rows
  // of a section before moving on to the next so the number of sections can
  // be estimated as 'numVertices / numRows'.
  auto exchangeCost = ceilDiv(maxVerticesPerTile, numRows) *
                      ceilDiv(numCols, split) * sizeof(int) /
                      exchangeBytesPerCycle;
  // After the partial sums are calculated, a second compute set will
  // calculate the reduction with a vertex for each element of the output (i.e
  // 'numRows' elements). The compute cost for the reduction can be calculated
  // with similar logic to above.
  auto maxReduceVerticesPerTile = std::max(ceilDiv(numRows, numTiles), 1U);
  auto cyclesPerReduceVertex = 5 + (split + 1) / 2;
  auto reduceComputeCost = ceilDiv(maxReduceVerticesPerTile, numWorkers) *
                           numWorkers * cyclesPerReduceVertex;
  // Each tile reducing will have to receive 'split' number of elements per
  // set of partial sums it is reducing.
  auto reduceExchangeCost =
      maxReduceVerticesPerTile * split * sizeof(int) / exchangeBytesPerCycle;
  // The estimated total number of cycles it the sum of the compute and
  // exchange cycles for the dot product and reduce phases.
  auto totalCost =
      computeCost + exchangeCost + reduceComputeCost + reduceExchangeCost;
  if (verbose) {
    std::cout << "colsAxisSplit=" << split << ", total cost=" << totalCost
              << " (compute cost=" << computeCost
              << ", exchange cost=" << exchangeCost
              << ", reduce exchange cost=" << reduceExchangeCost
              << ", reduce compute cost=" << reduceComputeCost << ")\n";
  }
  return totalCost;
}

// Given a specific number of rows and columns, calculate the optimal
// column axis split to use.
unsigned calcOptimalColAxisSplit(const Graph &graph, unsigned numRows,
                                 unsigned numCols) {
  unsigned best = 1;
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  unsigned worstCost = std::numeric_limits<unsigned>::min();
  for (unsigned split = 1; split < numCols; ++split) {
    auto cost =
        estimateCycles(graph, numRows, numCols, split, VERBOSE_LEVEL >= 2);
    if (bestCost > cost) {
      best = split;
      bestCost = cost;
    }
    worstCost = std::max(cost, worstCost);
  }
  if (VERBOSE_LEVEL >= 1) {
    std::cout << "Best split chosen:\n";
    estimateCycles(graph, numRows, numCols, best, true);
    std::cout << "Worst cost seen: " << worstCost << "\n";
  }
  return best;
}

// This function returns a device side program that will find the hamming distance of
// the data in the 2-d tensor 'matrix' with the 1-d vector held
// in the 'in' tensor. When the program executes
// the result is placed in the 'out' 1-d tensor.
Program buildMultiplyProgram(Graph &graph, Tensor matrix, Tensor in,
                             Tensor out) {
  auto numRows = matrix.dim(0);
  auto numCols = matrix.dim(1);
  // Get the optimal column axis split to split the number of columns
  // into partial sums.
  unsigned colAxisSplit = calcOptimalColAxisSplit(graph, numRows, numCols);

  // Create a tensor to hold the intermediate calculated partial sums.
  auto partials = graph.addVariable(INT, {numRows, colAxisSplit}, "partials");

  const auto numTiles = graph.getTarget().getNumTiles();

  // The input vector is used by all tiles. So their is no obvious place
  // to put it that reduces communications. To balance memory, just
  // spread it over all the tiles.
  for (unsigned i = 0; i < numCols; ++i) {
    graph.setTileMapping(in[i], i * numTiles / numCols);
  }

  // Create a compute set to hold the vertices to perform the
  // partial sum calculations.
  ComputeSet mulCS = graph.addComputeSet("mulCS");

  // Create a vertex for each segment, for each row.
  for (unsigned i = 0; i < colAxisSplit; ++i) {
    // The split may not divide the number of columns exactly. So the
    // columns in this segment need to be quantized.
    unsigned beginCol = (i * numCols) / colAxisSplit;
    unsigned endCol = ((i + 1) * numCols) / colAxisSplit;
    if (beginCol == endCol)
      continue;
    for (unsigned row = 0; row < numRows; ++row) {
      // The matrix elements for the dot product are the slice of the
      // row between 'beginCol' and 'endCol'.
      auto matrixElements = matrix[row].slice(beginCol, endCol);
      // The input elements for the dot product are just the slice of the
      // input vector between 'beginCol' and 'endCol'.
      auto inputElements = in.slice(beginCol, endCol);
      // Create a 'HamDistanceVertex' vertex in the 'mulCS' compute set and
      // connect the inputs 'a' and 'b' to the matrix and input elements to
      // be operated. Connect the output 'out' to an element of the
      // tensor 'partials'
      auto v = graph.addVertex(mulCS, "HamDistanceVertex",
                               {{"a", matrixElements},
                                {"b", inputElements},
                                {"out", partials[row][i]}});

      // This vertices are evenly spread over the tiles.
      unsigned tile = (i * numRows + row) * numTiles / (colAxisSplit * numRows);
      // Map the vertex and its associated data to the current tile.
      graph.setTileMapping(v, tile);
      graph.setTileMapping(matrix[row].slice(beginCol, endCol), tile);
      graph.setTileMapping(partials[row][i], tile);
      // Guess 5 cycles for the overhead of starting the vertex and setting
      // up the loop.
      // The hamming distance operation can be performed by vectorized mac instructions that
      // can perform 2 macs per cycle.
      graph.setPerfEstimate(v, 5 + ((endCol - beginCol) + 1) / 2);
    }
  }
  // Create a compute set to calculate the reduction.
  auto reduceCS = graph.addComputeSet("reduceCS");

  // For each output element create a vertex.
  for (unsigned row = 0; row < numRows; ++row) {
    // Create a 'ReduceVertex' that takes all the partial sums for a row
    // and reduces them to a single value placed in the output vector.
    auto v = graph.addVertex(reduceCS, "ReduceVertex",
                             {{"in", partials[row]}, {"out", out[row]}});
    // Map the computation to a tile such that the vertices are spread
    // evenly over the tiles. Map associated tensor data to the same tile.
    auto tile = (row * numTiles) / numRows;
    graph.setTileMapping(v, tile);
    graph.setTileMapping(out[row], tile);
    // Addition can be vectorized at 2 float additions per cycle.
    graph.setPerfEstimate(v, 5 + (partials[row].numElements() + 1) / 2);
  }

  // The program to perform the multiplication consists of executing the
  // compute set that calculates the partial sums followed by the compute
  // set that performs the reduction.
  return Sequence({Execute(mulCS), Execute(reduceCS)});
}

void help(const char *app) {
  std::cerr << "usage: " << app
            << " numRows numCols --device {model-ipu1,model-ipu2,ipu}\n";
}

int parse_args(unsigned &numRows, unsigned &numCols, const char *&dev, int argc,
               char **argv) {
  for (int a = 1; a < argc; ++a) {
    if (strcmp(argv[a], "--device") == 0) {
      ++a;
      if (a >= argc) {
        printf("Missing argument following --device\n");
        return -1;
      }
      dev = argv[a];
      if (strcmp(dev, "ipu") && strcmp(dev, "model-ipu1") &&
          strcmp(dev, "model-ipu2")) {
        printf("Unrecognised device %s\n", dev);
        return -1;
      }
    } else if (numRows == 0) {
      numRows = atoi(argv[a]);
      if (numRows <= 0) {
        printf("Malformed numRows argument\n");
        return -1;
      }
    } else if (numCols == 0) {
      numCols = atoi(argv[a]);
      if (numCols <= 0) {
        printf("Malformed numCols argument\n");
        return -1;
      }
    } else {
      printf("Unexpected arguments\n");
      return -1;
    }
  }
  if (numRows <= 0 || numCols <= 0) {
    printf("Missing rows/cols arguments\n");
    return -1;
  }
  return 0;
}

int main(int argc, char **argv) {
  unsigned numRows = 0;
  unsigned numCols = 0;
  const char *dev = "model-ipu2";

  int ret = parse_args(numRows, numCols, dev, argc, argv);
  if (ret != 0) {
    printf("Failed to parse arguments\n");
    help(argv[0]);
    return ret;
  }

  printf("Device %s\n", dev);

  std::cout << "2D array " << numRows << "x" << numCols
            << "HD by vector of size " << numCols << "\n";

  std::cout << "Constructing compute graph and control program\n";
  // This graph is going to target an simulated IPU. For simplicity in
  // this example the configuration of the exchange in this simulated model
  // is set to be more simplistic to reduce some latencies/delays in
  // the exchange fabric.

  Device device;

  if (strcmp(dev, "ipu") == 0) {
    // The DeviceManager is used to discover IPU devices
    auto manager = DeviceManager::createDeviceManager();

    // Attempt to attach to a single IPU:
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
    std::cout << "Trying to attach to IPU\n";
    auto it = std::find_if(devices.begin(), devices.end(),
                           [](Device &device) { return device.attach(); });

    if (it == devices.end()) {
      std::cerr << "Error attaching to device\n";
      return 1; // EXIT_FAILURE
    }

    device = std::move(*it);
    std::cout << "Attached to IPU " << device.getId() << std::endl;
  } else {
    char ipuVersion[] = "ipu1";
    strncpy(ipuVersion, &dev[6], strlen(ipuVersion));
    IPUModel ipuModel(ipuVersion);
    ipuModel.minIPUSyncDelay = 0;
    ipuModel.relativeSyncDelay = IPUModel::RelativeSyncDelayType::NO_DELAY;
    device = ipuModel.createDevice();
  }

  Graph graph(device);
  graph.addCodelets("hamdist_codelets.cpp");

  // Create tensors in the graph to hold the input/output data.
  Tensor matrix = graph.addVariable(INT, {numRows, numCols}, "matrix");
  Tensor inputVector = graph.addVariable(INT, {numCols}, "inputVector");
  Tensor outputVector = graph.addVariable(INT, {numRows}, "outputVector");

  // Create host buffers for the inputs and outputs and fill the inputs
  // with sample data.
  auto hMatrix = std::vector<int>(numRows * numCols);
  auto hInput = std::vector<int>(numCols);
  auto hOutput = std::vector<int>(numRows);
    
    // Fill matrix and vector with random data 0-255
    srand(time(0));
  for (unsigned col = 0; col < numCols; col++) {
    hInput[col] = rand()%255;
    for (unsigned row = 0; row < numRows; row++) {
      hMatrix[row * numCols + col] = rand()%255;
    }
  }    

  // Create a device program to multiply two tensors together.
  auto mulProg = buildMultiplyProgram(graph, matrix, inputVector, outputVector);

  // Set up data streams to copy data in and out of graph
  auto inStreamV = graph.addHostToDeviceFIFO("inputVector", INT, numCols);
  auto inStreamM =
      graph.addHostToDeviceFIFO("inputMatrix", INT, numCols * numRows);
  auto outStream = graph.addDeviceToHostFIFO("out", INT, numRows);

  // Write graph and execution profiles to files
  OptionFlags engineOpts{{"autoReport.all", "true"}};

  // Create an engine from the compute graph and control program.
  Engine engine(graph,
                Sequence({Copy(inStreamV, inputVector), Copy(inStreamM, matrix),
                          mulProg, Copy(outputVector, outStream)}),
                engineOpts);
  engine.load(device);
  engine.connectStream("inputVector", hInput.data());
  engine.connectStream("inputMatrix", hMatrix.data());
  engine.connectStream("out", hOutput.data());


    
    
      // Execute the program
  std::cout << "Running graph program to find the hamming distance of matrix and vector\n";
    auto start = std::chrono::high_resolution_clock::now();
  engine.run();
 auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    
//     std::cout<<"Print output:"<<std::endl;
    
//     for(unsigned i = 0; i<numRows; i++){
     
//         std::cout<<hOutput[i]<<std::endl;
//     }
    
    std::cout<<std::endl;
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;
    


  // Product a report showing the profile of execution on the simulated IPU.
  // engine.printProfileSummary(std::cout,
  //                            OptionFlags{{"showExecutionSteps", "true"}});
    
    
  return 0;
}