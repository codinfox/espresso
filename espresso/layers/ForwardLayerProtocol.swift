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
   Forward methods can take optional parameters. For the input layers, the parameters can be nil. `forward` method is a wrapper of `forward_cpu` and `forward_gpu`. `forward` method is public in the implementing classes, while `forward_cpu` and `forward_gpu` are internal
   */
  func forward(bottomOpt: [Tensor]?)
  func forward_cpu(bottomOpt: [Tensor]?)
  func forward_gpu(bottomOpt: [Tensor]?)

  /**
   Reshape the current output (gradient maybe) to conform to the output dimension of the bottom layer
   
   Always check first. It may not be necessary to reshape if it has already been same with the target shape

   - parameter bottomDimensions: The dimension of the bottom layer
   */
  func reshape(bottomDimensionsOpt: [Int]?)
}

extension ForwardLayerProtocol {
  func forward(bottom: [Tensor]?) {
    if isCpu {
      forward_cpu(bottom)
    } else {
      forward_gpu(bottom)
    }
  }

  func forward_gpu(bottom: [Tensor]?) {
    forward_cpu(bottom)
  }

}