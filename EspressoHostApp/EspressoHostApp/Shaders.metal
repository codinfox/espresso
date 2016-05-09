//
//  Shaders.metal
//  espresso
//
//  Created by Jerry Zhang on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;

/* Data Structures */
struct MetalConvolutionParameter {
  int count;
  int padSize;
  int kernelSize;
  int stride;
  int inputChannel;
  int inputHeight;
  int inputWidth;
  int outputChannel;
  int outputHeight;
  int outputWidth;
};

struct MetalPoolingParameter {
  int count;
  int padSize;
  int kernelSize;
  int stride;
  int inputChannel;
  int inputHeight;
  int inputWidth;
  int outputChannel;
  int outputHeight;
  int outputWidth;
};

struct MetalReluParameter {
  int count;
  float negativeSlope;
};

struct MetalFullyConnectedParameter {
  int count;
  int numOutput;
  int numElementsPerBatch;
};

struct MetalSoftmaxParameter {
  int count;
  int numOutput;
  int mapSizeToPerformOn;
};

struct MetalSoftmaxWithLossParameter {

};

struct MetalDropoutParameter {

};

struct MetalLrnParameter {

};

struct MetalIm2colParameter {
  int count;
  int inputOffset;
  int kernelSize;
  int padSize;
  int stride;
  int inputChannels;
  int inputHeight;
  int inputWidth;
  int outputHeight;
  int outputWidth;
};

struct MetalSgemmParameter {
  uint count;
  bool transA;
  bool transB;
  uint m;
  uint n;
  uint k;
  float alpha;
  float beta;
};


/* Functions */
kernel void convolutionForward(const device float *input [[ buffer(0) ]],
                               device float *output [[ buffer(1) ]],
                               const device float *weights [[ buffer(2) ]],
                               const device float *bias [[ buffer(3) ]],
                               const device MetalConvolutionParameter *convolvParams [[ buffer(4) ]],
                               uint id [[ thread_position_in_grid ]]) {
  //count = batchSize * outputChannel * outputHeight * outputWidth
  if (id >= (uint)convolvParams[0].count) {
    return;
  }
  int padSize = convolvParams[0].padSize;
  int kernelSize = convolvParams[0].kernelSize;
  int stride = convolvParams[0].stride;

  int inputChannel = convolvParams[0].inputChannel;
  int inputHeight = convolvParams[0].inputHeight;
  int inputWidth = convolvParams[0].inputWidth;

  int outputChannel = convolvParams[0].outputChannel;
  int outputHeight = convolvParams[0].outputHeight;
  int outputWidth = convolvParams[0].outputWidth;

  int oneBatchElements = outputChannel * outputHeight * outputWidth;
  int oneChannelElements = outputHeight * outputWidth;
  int batchNo = id / oneBatchElements;
  int channelNo = (id % oneBatchElements) / oneChannelElements;
  int startHeight = ((id % oneBatchElements) % oneChannelElements) / outputWidth;
  int startWidth = ((id % oneBatchElements) % oneChannelElements) % outputWidth;

  int inputStartHeight = startHeight * stride;
  int inputStartWidth = startWidth * stride;
  int inputOffset = batchNo * inputChannel * inputHeight;// * inputWidth(multiply in use)
  int weightsOffset = channelNo * inputChannel ;//* kernelSize * kernelSize (multiply in use)

  thread float convolv = 0;
  for (int inputChanNo = 0; inputChanNo < inputChannel; inputChanNo++) {
    for (int i = 0; i < kernelSize; i++) {
      for (int j = 0; j < kernelSize; j++) {
        int row = inputStartHeight + i;
        int col = inputStartWidth + j;
        if (row >= padSize && (row < (inputHeight + padSize)) &&
            col >= padSize && (col < (inputWidth + padSize))) {
          convolv += input[(inputOffset + inputChanNo * inputHeight + (row - padSize)) * inputWidth + (col - padSize)]
          * weights[((weightsOffset + inputChanNo) * kernelSize + i) * kernelSize + j];
        }
      }
    }
  }

  output[id] = convolv + bias[channelNo];
}

kernel void poolingMaxForward(const device float *input [[ buffer(0) ]],
                              device float *output [[ buffer(1) ]],
                              const device MetalPoolingParameter *poolingParams [[ buffer(2) ]],
                              uint id [[ thread_position_in_grid ]]) {
  if (id >= (uint)poolingParams[0].count) {
    return;
  }
  int padSize = poolingParams[0].padSize;
  int kernelSize = poolingParams[0].kernelSize;
  int stride = poolingParams[0].stride;

  int inputChannel = poolingParams[0].inputChannel;
  int inputHeight = poolingParams[0].inputHeight;
  int inputWidth = poolingParams[0].inputWidth;

  int outputChannel = poolingParams[0].outputChannel;
  int outputHeight = poolingParams[0].outputHeight;
  int outputWidth = poolingParams[0].outputWidth;

  int oneBatchElements = outputChannel * outputHeight * outputWidth;
  int oneChannelElements = outputHeight * outputWidth;
  int batchNo = id / oneBatchElements;
  int channelNo = (id % oneBatchElements) / oneChannelElements;
  int startHeight = ((id % oneBatchElements) % oneChannelElements) / outputWidth;
  int startWidth = ((id % oneBatchElements) % oneChannelElements) % outputWidth;

  int inputStartHeight = startHeight * stride;
  int inputStartWidth = startWidth * stride;

  int inputOffset = (batchNo * inputChannel + channelNo) * inputHeight * inputWidth;
  thread float pooled = -1000000000;//-FLT_MAX;
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      int row = inputStartHeight + i;
      int col = inputStartWidth + j;
      if (row >= padSize && (row < (inputHeight + padSize)) &&
          col >= padSize && (col < (inputWidth + padSize))) {
        pooled = max(pooled, input[inputOffset + (row - padSize) * inputWidth + (col - padSize)]);
      } else {
        pooled = max(pooled, 0.0);
      }
    }
  }
  output[id] = pooled;
}

kernel void poolingAvgForward(const device float *input [[ buffer(0) ]],
                              device float *output [[ buffer(1) ]],
                              const device MetalPoolingParameter *poolingParams [[ buffer(2) ]],
                              uint id [[ thread_position_in_grid ]]) {
  if (id >= (uint)poolingParams[0].count) {
    return;
  }
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

  int oneBatchElements = outputChannel * outputHeight * outputWidth;
  int oneChannelElements = outputHeight * outputWidth;
  int batchNo = id / oneBatchElements;
  int channelNo = (id % oneBatchElements) / oneChannelElements;
  int startHeight = ((id % oneBatchElements) % oneChannelElements) / outputWidth;
  int startWidth = ((id % oneBatchElements) % oneChannelElements) % outputWidth;

  int inputStartHeight = startHeight * stride;
  int inputStartWidth = startWidth * stride;

  int inputOffset = (batchNo * inputChannel + channelNo) * inputHeight * inputWidth;

  thread float pooled = 0;
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      int row = inputStartHeight + i;
      int col = inputStartWidth + j;
      if (row >= padSize && (row < (inputHeight + padSize)) &&
          col >= padSize && (col < (inputWidth + padSize))) {
        pooled += input[inputOffset + (row - padSize) * inputWidth + (col - padSize)];
      }
    }
  }
  output[id] = (pooled / (float)(kernelSize * kernelSize));
}

kernel void reluForward(const device float *input [[ buffer(0) ]],
                        device float *output [[ buffer(1) ]],
                        const device MetalReluParameter *reluParameters [[ buffer(2) ]],
                        uint id [[ thread_position_in_grid ]]) {
  // count = batchSize * input dimension
  float negativeSlope = float(reluParameters[0].negativeSlope);
  output[id] = fmax(0.0, input[id]) + negativeSlope * fmin(0.0, input[id]);
}


kernel void fullyConnectedForward(const device float *input [[ buffer(0) ]],
                                  device float *output [[ buffer(1) ]],
                                  const device float *weights [[ buffer(2) ]],
                                  const device float *bias [[ buffer(3) ]],
                                  const device MetalFullyConnectedParameter *fcParams [[ buffer(4) ]],
                                  uint id [[ thread_position_in_grid ]]) {
  if (id >= (uint)fcParams[0].count) {
    return;
  }
  int numOutput = fcParams[0].numOutput;
  int numElementsPerBatch = fcParams[0].numElementsPerBatch;
  int curBatch = id / numOutput;
  int curOutput = id % numOutput;

  for (int i = 0; i < numElementsPerBatch; i++) {
    output[id] += input[curBatch * numElementsPerBatch + i] * weights[curOutput * numElementsPerBatch + i] * bias[id];
  }
}

kernel void softmaxForward(const device float *input [[ buffer(0) ]],
                           device float *output [[ buffer(1) ]],
                           const device MetalSoftmaxParameter *softmaxParam [[ buffer(2) ]],
                           uint id [[ thread_position_in_grid ]]) {
  // count = batchSize * height * width
  if (id >= (uint)softmaxParam[0].count) {
    return;
  }
  int numOutput = softmaxParam[0].numOutput;
  int mapSizeToPerformOn = softmaxParam[0].mapSizeToPerformOn;

  int curBatch = id / mapSizeToPerformOn;
  int gridIndex = id % mapSizeToPerformOn;
  float maxPixel = -FLT_MAX;

  int fixedOffset = curBatch * numOutput * mapSizeToPerformOn;

  for (int curChan = 0; curChan < numOutput; curChan++) {
    maxPixel = max(maxPixel, input[fixedOffset + curChan * mapSizeToPerformOn + gridIndex]);
  }

  float Z = 0;
  for (int curChan = 0; curChan < numOutput; curChan++) {
    output[fixedOffset + curChan * mapSizeToPerformOn + gridIndex] = exp(input[curBatch * numOutput * mapSizeToPerformOn + curChan * mapSizeToPerformOn + gridIndex] - maxPixel);
    Z += output[fixedOffset + curChan * mapSizeToPerformOn + gridIndex];
  }

  for (int curChan = 0; curChan < numOutput; curChan++) {
    output[fixedOffset + curChan * mapSizeToPerformOn + gridIndex] /= Z;
  }
}

kernel void im2colGpu(const device float *input [[ buffer(0) ]],
                      device float *output [[ buffer(1) ]],
                      const device MetalIm2colParameter* im2colParam [[ buffer(2) ]],
                      uint id [[ thread_position_in_grid ]]) {
  if (id >= (uint)im2colParam[0].count) {
    return;
  }
  int inputChannels = im2colParam[0].inputChannels;
  int inputHeight = im2colParam[0].inputHeight;
  int inputWidth = im2colParam[0].inputWidth;

  int kernelSize = im2colParam[0].kernelSize;
  int padSize = im2colParam[0].padSize;
  int stride = im2colParam[0].stride;
  int inputOffset = im2colParam[0].inputOffset;

  int outputHeight = im2colParam[0].outputHeight;
  int outputWidth = im2colParam[0].outputWidth;

  // not considering batch for now
  int numElementsPerChan = outputHeight * outputWidth * kernelSize * kernelSize;
  int curChan = id / numElementsPerChan;
  int kernId = (id % numElementsPerChan) / (outputHeight * outputWidth);
  int kernRow = kernId / kernelSize;
  int kernCol = kernId % kernelSize;

  int colLength = outputHeight * outputWidth;

  int inputRowOff = ((id % numElementsPerChan) % colLength) / outputWidth;
  int inputColOff = ((id % numElementsPerChan) % colLength) % outputWidth;

  int inputRow = kernRow - padSize + inputRowOff * stride;
  int inputCol = kernCol - padSize + inputColOff * stride;

  int inputChannelSize = inputHeight * inputWidth;

  if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
    output[id] = input[inputOffset + curChan * inputChannelSize + inputRow * inputWidth + inputCol];
  } else {
    output[id] = 0;
  }
}

#pragma mark Math Functions

/**
 A: m rows by k columns
 B: k rows by n columns
 C: m rows by n columns

 If trans is specified, then m,n,k represents A,B dimensions after transpose (that is, really used in multiplication)
 */
kernel void espressoSgemm(const device float * A [[ buffer(0) ]],
                          const device float * B [[ buffer(1) ]],
                          device float * C [[ buffer(2) ]],
                          const device MetalSgemmParameter* sgemmParam [[ buffer(3) ]],
                          uint id [[ thread_position_in_grid ]]) {
  if (id >= sgemmParam[0].count) {
    return;
  }
  bool transA = sgemmParam[0].transA;
  bool transB = sgemmParam[0].transB;
  uint m = sgemmParam[0].m;
  uint n = sgemmParam[0].n;
  uint k = sgemmParam[0].k;
  float alpha = sgemmParam[0].alpha;
  float beta = sgemmParam[0].beta;

  uint aStep = transA ? m : 1;
  uint bStep = transB ? 1 : n;

  uint cRow = id / n, cCol = id % n;

  uint aOffset = transA ? (cRow) : (k * cRow);
  uint bOffset = transB ? (k * cCol) : (cCol);

  thread float result = 0;
  for (uint i = 0; i < k; i ++) {
    result += A[aOffset + i * aStep] * B[bOffset + i * bStep];
  }
  C[id] = alpha * result + beta * C[id];
}
