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
public class DropoutLayer: ForwardLayerProtocol, BackwardLayerProtocol, TrainableLayerProtocol {
  public var name : String
  public var dependencies : [String]
  public var output: Tensor = Tensor()
  public var gradient: Tensor = Tensor()
  public var weight: Tensor
  public var bias: Tensor
  public var engine: NetworkProperties.NetworkEngine

  var parameters : DropoutParameters

  func forwardCPU(bottomOpt: [Tensor]) {}
  func forwardGPU(bottomOpt: [Tensor]) {}

  func backwardCPU(topOpt: [Tensor]) {}
  func backwardGPU(topOpt: [Tensor]) {}

  func initWeights() {
  }

  func updateWeights(weightGrad: Tensor){
  }

  func initBias() {}

  func updateBias(biasGrad: Tensor) {
  }

  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {

  }

  public func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                                bottomDimensions: [[Int]]?) {
  }

  public init(parameters: DropoutParameters) {
    self.name = parameters.name
    self.parameters = parameters
    self.weight = Tensor(dimensions: [])
    self.bias = Tensor(dimensions: [])
    self.engine = .CPU
  }
}

public struct DropoutParameters : LayerParameterProtocol {
  public var name: String
  public let dropoutRatio : Float
  public var dependencies: [String]
  public init(name: String = "Dropout Layer", dependencies: [String], dropoutRatio: Float = 0.5) {
    self.name = name
    self.dropoutRatio = dropoutRatio
    self.dependencies = dependencies
  }
}