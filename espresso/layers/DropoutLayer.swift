//
//  DropoutLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Dropout layer.
 */
public class DropoutLayer: ForwardBackwardLayerProtocol {
  public var name : String
  public var isCpu: Bool
  public var output: [Tensor]
  public var gradient: [Tensor]
  public var weight: Tensor
  public var bias: Tensor

  var parameters : DropoutParameters

  func forward_cpu(bottomOpt: [Tensor]?) {}
  func forward_gpu(bottomOpt: [Tensor]?) {}

  func backward_cpu(topOpt: [Tensor]?) {}
  func backward_gpu(topOpt: [Tensor]?) {}

  func reshape(bottomDimensionsOpt: [Int]?) {
    // Reshape the output (and gradient)
  }

  func initWeights() {
  }

  func updateWeights(weightGrad: Tensor){
  }

  func initBias() {}

  func updateBias(biasGrad: Tensor) {
  }


  public init(name: String = "dropout", parameters: DropoutParameters) {
    self.name = name
    self.parameters = parameters
  }
  // Implement protocols
}

public struct DropoutParameters : LayerParameterProtocol {
  public let dropoutRatio : Float
  public init(dropoutRatio: Float = 0.5) {
    self.dropoutRatio = dropoutRatio
  }
}