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
  float negativeSlope = float(reluParameters.negativeSlope)
  output[id] = max(0, input[id]) + negativeSlope * min(0, input[id])
}