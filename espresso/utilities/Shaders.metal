//
//  Shaders.metal
//  espresso
//
//  Created by Jerry Zhang on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include <metal_stdlib>
#include <math.h>
using namespace metal;

/* Data Structures */
struct MetalConvolutionParameter {
  int padSize;
  int kernelSize;
  int stride;
  int inputChannel;
  int inputHeight;
  int inputWidth;
  int outputChannel;
  int outputHeight;
  int outputWidth;
}

struct MetalPoolingParameter {
  int padSize;
  int stride;
  int inputChannel;
  int inputHeight;
  int inputWidth;
  int outputChannel;
  int outputHeight;
  int outputWidth;
}

struct MetalReluParameter {
  float negativeSlope;
}

struct MetalFullyConnectedParameter {
  int numNeurons;
  int channel;
  int height;
  int width;
}

struct MetalSoftmaxParameter {
  int height;
  int width;
}

struct MetalSoftmaxWithLossParameter {

}

struct MetalDropoutParameter {

}

struct MetalLrnParameter {
  
}


/* Functions */
kernel void convolutionForward(const device float *input [[ buffer(0) ]],
                               const device float *output [[ buffer(1) ]],
                               const device float *kernels [[ buffer(2) ]],
                               const MetalConvolutionParameter *convolvParams [[ buffer(3) ]],
                               const id [[ thread_position_in_grid ]]) {
  int padSize = convolvParams[0].padSize;
  int kernelSize = convolvParams[0].kernelSize;
  int stride = convolvParams[0].stride;

  int inputChannel = convolvParams[0].inputChannel;
  int inputHeight = convolvParams[0].inputHeight;
  int inputWidth = convolvParams[0].inputWidth;

  int outputChannel = convolvParams[0].outputChannel;
  int outputHeight = convolvParams[0].outputHeight;
  int outputWidth = convolvParams[0].outputWidth;

  int channelNo = id / (outputChannel * outputHeight * outputWidth);
  int startHeight = id / (outputChannel * outputWidth);
  int startWidth = (id / outputChannel) % outputWidth;

  int inputStartHeight = startHeight * stride;
  int inputStartWidth = startWidth * stride;
  /* output should be initialized to 0 */
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      int row = inputStartHeight + i;
      int col = inputStartWidth + j;
      if (row >= padSize && row < inputHeight - padSize &&
          col >= padSize && col < inputWidth - padSize) {
        output[id] += input[channelNo * inputHeight * inputWidth + row * inputWidth + col]
        * kernels[channelNo * kernelSize * kernelSize + i * kernelSize + j];
      }
    }
  }

}

kernel void pollingMaxForward(const device float *input [[ buffer(0) ]],
                              const device float *output [[ buffer(1) ]],
                              const MetalPollingParameter *poolingParams [[ buffer(2) ]],
                              const id [[ thread_position_in_grid ]]) {
  int padSize = poolingParams[0].padSize;
  int kernelSize = poolingParams[0].kernelSize;
  int stride = poolingParams[0].stride;

  int inputChannel = poolingParams[0].inputChannel;
  int inputHeight = poolingParams[0].inputHeight;
  int inputWidth = poolingParams[0].inputWidth;

  int outputChannel = poolingParams[0].outputChannel;
  int outputHeight = poolingParams[0].outputHeight;
  int outputWidth = poolingParams[0].outputWidth;

  int channelNo = id / (outputChannel * outputHeight * outputWidth);
  int startHeight = id / (outputChannel * outputWidth);
  int startWidth = (id / outputChannel) % outputWidth;

  int inputStartHeight = startHeight * stride;
  int inputStartWidth = startWidth * stride;
  output[id] = -INFINITY;
  /* output should be initialized to 0 */
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      int row = inputStartHeight + i;
      int col = inputStartWidth + j;
      if (row >= padSize && row < inputHeight - padSize &&
          col >= padSize && col < inputWidth - padSize) {
        output[id] = max(output[id], input[channelNo * inputHeight * inputWidth + row * inputWidth + col]);
      }
    }
  }
}

kernel void pollingAvgForward(const device float *input [[ buffer(0) ]],
                              const device float *output [[ buffer(1) ]],
                              const MetalPollingParameter *poolingParams [[ buffer(2) ]],
                              const id [[ thread_position_in_grid ]]) {
  // count = batchSize * outputChannel * outputHeight * outputWidth
  int padSize = poolingParams[0].padSize;
  int kernelSize = poolingParams[0].kernelSize;
  int stride = poolingParams[0].stride;

  int inputChannel = poolingParams[0].inputChannel;
  int inputHeight = poolingParams[0].inputHeight;
  int inputWidth = poolingParams[0].inputWidth;

  int outputChannel = poolingParams[0].outputChannel;
  int outputHeight = poolingParams[0].outputHeight;
  int outputWidth = poolingParams[0].outputWidth;

  // need to verify the correctness of these values
  int batchNo = id / (outputChannel * outputHeight * outputWidth);
  int withinBatch = id % (outputChannel * outputHeight * outputWidth);
  int channelNo = withinBatch / (outputHeight * outputWidth);
  int withinChannel = withinBatch % (outputHeight * outputWidth);
  int startHeight = withinChannel / outputWidth;
  int startWidth = withinChannel % outputWidth;

  int inputStartHeight = startHeight * stride;
  int inputStartWidth = startWidth * stride;
  output[id] = 0;
  int count = 0;
  /* output should be initialized to 0 */
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      int row = inputStartHeight + i;
      int col = inputStartWidth + j;
      if (row >= padSize && row < inputHeight - padSize &&
          col >= padSize && col < inputWidth - padSize) {
        output[id] += input[batchNo * inputChannel * inputHeight * inputWidth + channelNo * inputHeight * inputWidth + row * inputWidth + col];
        count += 1;
      }
    }
  }
  output[id] /= count;
}

kernel void reluForward(const device float *input [[ buffer(0) ]],
                        const device flaot *output [[ buffer(1) ]],
                        const device MetalReluParameter *reluParameters [[ buffer(2) ]],
                        uint id [[ thread_position_in_grid ]]) {
  // count = batchSize * input dimension
  float negativeSlope = float(reluParameters[0].negativeSlope)
  output[id] = max(0, input[id]) + negativeSlope * min(0, input[id])
}


kernel void fullyConnectedForward(const device float *input [[ buffer(0) ]],
                                  const device flaot *output [[ buffer(1) ]],
                                  const device float *weights [[ buffer(2) ]],
                                  const device float *bias [[ buffer(3) ]],
                                  const device MetalFullyConnectedParameter *fcParams [[ buffer(4) ]],
                                  uint id [[ thread_position_in_grid ]]) {
  // count = batchSize * numNeurons * chnnels * height * width
  // maybe change count to batchSize * numNeurons to avoid concurrent access to output?
  int numNeurons = fcParams[0].numNeurons;
  int channel = fcParams[0].channel;
  int height = fcParams[0].height;
  int width = fcParams[0].width;

  int batchNo = id / (numNeurons * channel * height * width);
  int withinBatch = id % (numNeurons * channel * height * width);
  int neuronNo = withinBatch / (channel * height * width);
  int withinNeuron = withinBatch % (channel * height * width);
  int channelNo = withinNeuron / (height * width);
  int withinChannel = withinNeuron % (height * width);
  int heightNo = withinChannel / width;
  int widthNo = withinChannel % width;
  output[batchNo * numNeurons + neuronNo] += input[batchNo * channel * height * width + channelNo * height * width + heightNo * width + widthNo] * weights[id] * bias[id];
}

kernel void softmaxForward(const device float *input [[ buffer(0) ]],
                           const device flaot *output [[ buffer(1) ]],
                           const device MetalSoftmaxParameter *softmaxParam [[ buffer(2) ]],
                           uint id [[ thread_position_in_grid ]]) {
  // count = batchSize * channels
  int height = softmaxParam[0].height;
  int width = softmaxParam[0].width;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      output[id] += exp(-input[id * height * width + i * width + j])
    }
  }
}