//
//  DataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

typealias ForwardLayerMethodType = ([Tensor]?)->()
/** @brief Protocol for Forward Layers
 */
protocol ForwardLayerProtocol : LayerProtocol {

  var output: Tensor { get set }
  var forwardMethod: ForwardLayerMethodType? { get set }

  /** To be called in feedforward pass.
   Forward methods can take optional parameters. For the input layers, the parameters can be nil. `forward` method is a wrapper of `forwardCPU` and `forwardGPU`. `forward` method is public in the implementing classes, while `forwardCPU` and `forwardGPU` are internal
   
   forward method does not need to implemented
   
   - parameter bottom: Some layer can accept multiple inputs, thus this is an array of tensors. All tensors should have the same batchSize
   */
  mutating func forward(bottom: [Tensor]?)
  mutating func forwardCPU(bottom: [Tensor]?)
  mutating func forwardGPU(bottom: [Tensor]?)

  /**
   Output dimensions
   TODO: Can it be multiple outputs?

   - returns: <#return value description#>
   */
  func outputDimensions() -> [[Int]]

  /**
   Reshape the current output (gradient maybe) to conform to the output dimension of the bottom layer
   
   Always check first. It may not be necessary to reshape if it has already been same with the target shape
   
   The first dimension is the batch size

   - parameter bottomDimensions: The dimension of the bottom layer
   */
  mutating func reshapeByBottomDimensions(bottomDimensions: [[Int]])
}

extension ForwardLayerProtocol {
  mutating func forward(bottom: [Tensor]?) {
    self.forwardMethod!(bottom)
  }

  func outputDimensions() -> [[Int]] {
    return [output.dimensions]
  }
}