//
//  MetalParameters.swift
//  espresso
//
//  Created by Jerry Zhang on 4/18/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

public struct MetalReluParameter {
  var negativeSlope: Float
  var inputBatchSize: Int
  var inputChannels: Int
  var inputHeight: Int
  var inputWidht: Int
  init(negativeSlope: Float, inputDim: [Int]) {
    self.negativeSlope = negativeSlope
    self.inputBatchSize = inputDim[0]
    self.inputChannels = inputDim[1]
    self.inputHeight = inputDim[2]
    self.inputWidht = inputDim[3]
  }
}