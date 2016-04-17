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
  public var output: [Tensor]
  public var gradient: [Tensor]
  public var weight: Tensor
  public var bias: Tensor
  public var engine: NetworkProperties.NetworkEngine

  var parameters : DropoutParameters

  func forwardCPU(bottomOpt: [Tensor]?) {}
  func forwardGPU(bottomOpt: [Tensor]?) {}

  func backwardCPU(topOpt: [Tensor]?) {}
  func backwardGPU(topOpt: [Tensor]?) {}

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

  public func layerSetUp(networkProperties: NetworkProperties) {
  }

  public init(name: String = "dropout", parameters: DropoutParameters) {
    self.name = name
    self.parameters = parameters
    self.output = []
    self.gradient = [] // Not initialized, needs to be resized
    self.weight = Tensor(dimensions: [])
    self.bias = Tensor(dimensions: [])
    self.engine = .CPU
  }
}

public struct DropoutParameters : LayerParameterProtocol {
  public let dropoutRatio : Float
  public init(dropoutRatio: Float = 0.5) {
    self.dropoutRatio = dropoutRatio
  }
}