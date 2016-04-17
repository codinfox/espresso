//
//  LrnLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief LRN layer.
 */
public class LRNLayer: ForwardBackwardLayerProtocol {
  public var name : String
  public var output: [Tensor]
  public var gradient: [Tensor]
  public var weight: Tensor
  public var bias: Tensor
  public var engine: NetworkProperties.NetworkEngine

  var parameters : LRNParameters

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

  public init(name: String = "lrn", parameters: LRNParameters) {
    self.name = name
    self.parameters = parameters
    self.output = []
    self.gradient = [] // Not initialized, needs to be resized
    self.weight = Tensor(dimensions: [])
    self.bias = Tensor(dimensions: [])
    self.engine = .CPU
  }
}

public struct LRNParameters : LayerParameterProtocol {

  public enum NormRegion {
    case ACROSS_CHANNELS
    case WITHIN_CHANNEL
  }

  public let localSize: Int
  public let alpha: Float
  public let beta: Float
  public let region: NormRegion

  public init(localSize: Int = 5,
              alpha: Float = 1,
              beta: Float = 0.75,
              region: NormRegion = .ACROSS_CHANNELS) {
    self.localSize = localSize
    self.alpha = alpha
    self.beta = beta
    self.region = region
  }
}