//
//  DataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Protocol for Forward Layers
 */
protocol ForwardLayerProtocol : LayerProtocol {
  var output: [Tensor] { get set }

  /** To be called in feedforward pass.
   Forward methods can take optional parameters. For the input layers, the parameters can be nil. `forward` method is a wrapper of `forwardCPU` and `forwardGPU`. `forward` method is public in the implementing classes, while `forwardCPU` and `forwardGPU` are internal
   */
  func forward(bottomOpt: [Tensor]?)
  func forwardCPU(bottomOpt: [Tensor]?)
  func forwardGPU(bottomOpt: [Tensor]?)

  /**
   Reshape the current output (gradient maybe) to conform to the output dimension of the bottom layer
   
   Always check first. It may not be necessary to reshape if it has already been same with the target shape

   - parameter bottomDimensions: The dimension of the bottom layer
   */
  func reshape(bottomDimensionsOpt: [Int]?)
}

extension ForwardLayerProtocol {
  func forward(bottomOpt: [Tensor]?) {
    switch engine {
    case .CPU:
      forwardCPU(bottomOpt)
    case .GPU:
      forwardGPU(bottomOpt)
    }
  }

  func forwardGPU(bottomOpt: [Tensor]?) {
    forwardCPU(bottomOpt)
  }

}