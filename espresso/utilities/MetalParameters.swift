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
  init(negativeSlope: Float, inputDim: [Int]) {
    self.negativeSlope = negativeSlope
  }
}