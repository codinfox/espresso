////
////  Shaders.metal
////  espresso
////
////  Created by Jerry Zhang on 4/17/16.
////  Copyright © 2016 CMU. All rights reserved.
////
//
//#include <metal_stdlib>
//#include <math.h>
//using namespace metal;
//
///* Data Structures */
//struct MetalConvolutionParameter {
//  int padSize;
//  int kernelSize;
//  int stride;
//  int inputChannel;
//  int inputHeight;
//  int inputWidth;
//  int outputChannel;
//  int outputHeight;
//  int outputWidth;
//}
//
//struct MetalPoolingParameter {
//  int padSize;
//  int stride;
//  int inputChannel;
//  int inputHeight;
//  int inputWidth;
//  int outputChannel;
//  int outputHeight;
//  int outputWidth;
//}
//
//struct MetalReluParameter {
//  float negativeSlope;
//}
//
//struct MetalFullyConnectedParameter {
//  int numOutput;
//  int numElementsPerBatch;
//}
//
//struct MetalSoftmaxParameter {
//  int numOutput;
//  int totalNumberOfDistributions;
//  int mapSizeToPerformOn;
//}
//
//struct MetalSoftmaxWithLossParameter {
//
//}
//
//struct MetalDropoutParameter {
//
//}
//
//struct MetalLrnParameter {
//
//}
//
///* Functions */
//kernel void convolutionForward(const device float *input [[ buffer(0) ]],
//                               const device float *output [[ buffer(1) ]],
//                               const device float *weights [[ buffer(2) ]],
//                               const device float *bias [[ buffer(3) ]],
//                               const MetalConvolutionParameter *convolvParams [[ buffer(4) ]],
//                               const id [[ thread_position_in_grid ]]) {
//  int padSize = convolvParams[0].padSize;
//  int kernelSize = convolvParams[0].kernelSize;
//  int stride = convolvParams[0].stride;
//
//  int inputChannel = convolvParams[0].inputChannel;
//  int inputHeight = convolvParams[0].inputHeight;
//  int inputWidth = convolvParams[0].inputWidth;
//
//  int outputChannel = convolvParams[0].outputChannel;
//  int outputHeight = convolvParams[0].outputHeight;
//  int outputWidth = convolvParams[0].outputWidth;
//
//  int oneBatchElements = outputChannel * outputHeight * outputWidth;
//  int oneChannelElements = outputHeight * outputWidth;
//  int batchNo = id / oneBatchElements;
//  int channelNo = id % oneBatchElements / oneChannelElements;
//  int startHeight = id % oneChannelElements / outputWidth;
//  int startWidth = id % oneChannelElements % outputWidth;
//
//  int inputStartHeight = startHeight * stride;
//  int inputStartWidth = startWidth * stride;
//  int inputOffset = batchNo * inputChannel * inputHeight * inputWidth;
//
//  output[id] = 0
//  for (int inputChanNo = 0; inputChanNo < inputChannel; inputChanNo++) {
//    for (int i = 0; i < kernelSize; i++) {
//      for (int j = 0; j < kernelSize; j++) {
//        int row = inputStartHeight + i;
//        int col = inputStartWidth + j;
//        if (row >= padSize && row < inputHeight + padSize &&
//            col >= padSize && col < inputWidth + padSize) {
//          output[id] += input[inputOffset + inputChanNo * inputHeight * inputWidth + (row - padSize) * inputWidth + (col - padSize)]
//          * weights[inputChanNo * kernelSize * kernelSize + i * kernelSize + j];
//        }
//      }
//    }
//  }
//  output[id] += bias[channelNo]
//}
//
//kernel void poolingMaxForward(const device float *input [[ buffer(0) ]],
//                              const device float *output [[ buffer(1) ]],
//                              const MetalPollingParameter *poolingParams [[ buffer(2) ]],
//                              const id [[ thread_position_in_grid ]]) {
//  int padSize = poolingParams[0].padSize;
//  int kernelSize = poolingParams[0].kernelSize;
//  int stride = poolingParams[0].stride;
//
//  int inputChannel = poolingParams[0].inputChannel;
//  int inputHeight = poolingParams[0].inputHeight;
//  int inputWidth = poolingParams[0].inputWidth;
//
//  int outputChannel = poolingParams[0].outputChannel;
//  int outputHeight = poolingParams[0].outputHeight;
//  int outputWidth = poolingParams[0].outputWidth;
//
//  int oneBatchElements = outputChannel * outputHeight * outputWidth;
//  int oneChannelElements = outputHeight * outputWidth;
//  int batchNo = id / oneBatchElements;
//  int channelNo = id % oneBatchElements / oneChannelElements;
//  int startHeight = id % oneChannelElements / outputWidth;
//  int startWidth = id % oneChannelElements % outputWidth;
//
//  int inputStartHeight = startHeight * stride;
//  int inputStartWidth = startWidth * stride;
//
//  int inputOffset = batchNo * inputChannel * inputHeight * inputWidth + channelNo * inputHeight * inputWidth;
//  output[id] = -INFINITY;
//  for (int i = 0; i < kernelSize; i++) {
//    for (int j = 0; j < kernelSize; j++) {
//      int row = inputStartHeight + i;
//      int col = inputStartWidth + j;
//      if (row >= padSize && row < inputHeight - padSize &&
//          col >= padSize && col < inputWidth - padSize) {
//        output[id] = max(output[id], input[inputOffset + (row - padSize) * inputWidth + (col - padSize)]);
//      } else {
//        output[id] = max(output[id], 0)
//      }
//    }
//  }
//}
//
//kernel void poolingAvgForward(const device float *input [[ buffer(0) ]],
//                              const device float *output [[ buffer(1) ]],
//                              const MetalPollingParameter *poolingParams [[ buffer(2) ]],
//                              const id [[ thread_position_in_grid ]]) {
//  // count = batchSize * outputChannel * outputHeight * outputWidth
//  int padSize = poolingParams[0].padSize;
//  int kernelSize = poolingParams[0].kernelSize;
//  int stride = poolingParams[0].stride;
//
//  int inputChannel = poolingParams[0].inputChannel;
//  int inputHeight = poolingParams[0].inputHeight;
//  int inputWidth = poolingParams[0].inputWidth;
//
//  int outputChannel = poolingParams[0].outputChannel;
//  int outputHeight = poolingParams[0].outputHeight;
//  int outputWidth = poolingParams[0].outputWidth;
//
//  int oneBatchElements = outputChannel * outputHeight * outputWidth;
//  int oneChannelElements = outputHeight * outputWidth;
//  int batchNo = id / oneBatchElements;
//  int channelNo = id % oneBatchElements / oneChannelElements;
//  int startHeight = id % oneChannelElements / outputWidth;
//  int startWidth = id % oneChannelElements % outputWidth;
//
//  int inputStartHeight = startHeight * stride;
//  int inputStartWidth = startWidth * stride;
//
//  int count = 0
//  int inputOffset = batchNo * inputChannel * inputHeight * inputWidth + channelNo * inputHeight * inputWidth;
//
//  output[id] = 0;
//  for (int i = 0; i < kernelSize; i++) {
//    for (int j = 0; j < kernelSize; j++) {
//      int row = inputStartHeight + i;
//      int col = inputStartWidth + j;
//      if (row >= padSize && row < inputHeight - padSize &&
//          col >= padSize && col < inputWidth - padSize) {
//        output[id] += input[inputOffset + (row - padSize) * inputWidth + (col - padSize)];
//        count++;
//      }
//    }
//  }
//  if (count == 0) {
//    output[id] = 0
//  } else {
//    output[id] /= count;
//  }
//}
//
//kernel void reluForward(const device float *input [[ buffer(0) ]],
//                        const device float *output [[ buffer(1) ]],
//                        const device MetalReluParameter *reluParameters [[ buffer(2) ]],
//                        uint id [[ thread_position_in_grid ]]) {
//  // count = batchSize * input dimension
//  float negativeSlope = float(reluParameters[0].negativeSlope)
//  output[id] = max(0, input[id]) + negativeSlope * min(0, input[id])
//}
//
//
//kernel void fullyConnectedForward(const device float *input [[ buffer(0) ]],
//                                  const device float *output [[ buffer(1) ]],
//                                  const device float *weights [[ buffer(2) ]],
//                                  const device float *bias [[ buffer(3) ]],
//                                  const device MetalFullyConnectedParameter *fcParams [[ buffer(4) ]],
//                                  uint id [[ thread_position_in_grid ]]) {
//  int numOutput = fcParams[0].numOutput;
//  int numElementsPerBatch = fcParams[0].numElementsPerBatch;
//  int curBatch = id / numOutput;
//  int curOutput = id % numOutput;
//
//  for (int i = 0; i < numElementsPerBatch; i++) {
//    output[id] += input[curBatch * numElementsPerBatch + i] * weights[curOutput * numElementsPerBatch + i] * bias[id];
//  }
//}
//
//kernel void softmaxForward(const device float *input [[ buffer(0) ]],
//                           const device float *output [[ buffer(1) ]],
//                           const device MetalSoftmaxParameter *softmaxParam [[ buffer(2) ]],
//                           uint id [[ thread_position_in_grid ]]) {
//  // count = batchSize * height * width
//  int numOutput = softmaxParam[0].numOutput;
//  int totalNumberOfDistributions = softmaxParam[0].totalNumberOfDistributions;
//  int mapSizeToPerformOn = softmaxParam[0].mapSizeToPerformOn;
//
//  int curBatch = id / mapSizeToPerformOn;
//  int gridIndex = id % mapSizeToPerformOn;
//  float maxPixel = -INFINITY;
//
//  int fixedOffset = curBatch * numOutput * mapSizeToPerformOn;
//
//  for (int curChan = 0; curChan < numOutput; curChan++) {
//    maxPixel = max(maxPixel, input[fixedOffset + curChan * mapSizeToPerformOn + gridIndex]);
//  }
//
//  float Z = 0;
//  for (int curChan = 0; curChan < numOutput; curChan++) {
//      output[fixedOffset + curChan * mapSizeToPerformOn + gridIndex] = exp(input[curBatch * numOutput * mapSizeToPerformOn + curChan * mapSizeToPerformOn + gridIndex] - maxPixel)
//      Z += output[fixedOffset + curChan * mapSizeToPerformOn + gridIndex];
//  }
//
//  for (int curChan = 0; curChan < numOutput; curChan++) {
//    output[fixedOffset + curChan * mapSizeToPerformOn + gridIndex] /= Z;
//  }
//}