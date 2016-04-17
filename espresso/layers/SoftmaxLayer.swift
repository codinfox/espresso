//
//  Softmax.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Softmax layer.
 This can also be not backwardable
 */
public class SoftmaxLayer: ForwardBackwardLayerProtocol {
  public var name : String
  public var output: [Tensor]
  public var gradient: [Tensor]
  public var weight: Tensor
  public var bias: Tensor
  public var engine: NetworkProperties.NetworkEngine

  var parameters : SoftmaxParameters

  func forward_cpu(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0){
      let bottom = bottomOpt!
      let batchSize = bottom.count
      let channels = bottom[0].dimensions[0]
      let height = bottom[0].dimensions[1]
      let width = bottom[0].dimensions[2]
      for i in 0..<batchSize {
        output[i].reset(0)
      }
      for i in 0..<batchSize {
        for j in 0..<channels {
          for k in 0..<height {
            for l in 0..<width {
              output[i][j] += exp(-bottom[i][j,k,l])
            }
          }
        }
      }
    }
  }

  func forward_gpu(bottomOpt: [Tensor]?) {
  }

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

  public func layerSetUp(networkProperties: NetworkProperties) {
  }
  public init(name: String = "softmax", parameters: SoftmaxParameters) {
    self.name = name
    self.parameters = parameters
    self.output = []
    self.gradient = [] // Not initialized, needs to be resized
    self.weight = Tensor(dimensions: [])
    self.bias = Tensor(dimensions: [])
    self.engine = .CPU
  }
}

public struct SoftmaxParameters : LayerParameterProtocol {
  public var negativeSlope : Tensor.DataType
  public init(negativeSlope: Tensor.DataType = 0) {
    self.negativeSlope = negativeSlope
  }
}
