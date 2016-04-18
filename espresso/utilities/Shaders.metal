//
//  Shaders.metal
//  espresso
//
//  Created by Jerry Zhang on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


kernel void reluForward(const device float *input [[ buffer(0) ]],
                         const device flaot *output [[ buffer(1) ]],
                         const device MetalReluParameter reluParameters [[ buffer(2) ]],
                         uint id [[ thread_position_in_grid ]]) {
  int batchSize = int(reluParameters.inputBatchSize)
  int channels = int(reluParameters.inputChannels)
  int height = int(reluParameters.inputHeight)
  int width = int(reluParameters.inputWidth)
  float negativeSlope = float(reluParameters.negativeSlope)
  for i in 0..<batchSize {
    for j in 0..<channels {
      for k in 0..<height {
        for l in 0..<width {
          output[i][j,k,l] = max(0, bottom[i][j,k,l]) + negativeSlope * min(0, bottom[i][j,k,l])
        }
      }
    }
  }
}