//
//  TrainableLayerProtocol.swift
//  espresso
//
//  Created by Zhihao Li on 4/15/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

protocol TrainableLayerProtocol : LayerProtocol {
  var weights : Tensor { get set }
  var bias : Tensor { get set }

  mutating func initWeights()
  /** Take in the gradient of the weights, and update weights
   The update procedure should be conducted by the solver but, besides the global learning rate and other parameters, also consider the local learning rate and parameters (weight decay and etc.)
   */
  mutating func updateWeights(weightGrad: Tensor) // TODO

  mutating func initBias()

  mutating func updateBias(biasGrad: Tensor)
}

extension TrainableLayerProtocol {
  mutating func initWeights() {
    // Do nothing
  }
  mutating func updateWeights(weightGrad: Tensor) {
    // Do nothing
  }
  mutating func initBias() {
    // Do nothing
  }
  mutating func updateBias(biasGrad: Tensor) {
    // Do nothing
  }
}