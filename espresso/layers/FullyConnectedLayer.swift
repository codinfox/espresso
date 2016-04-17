//
//  FullyConnectedLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Fully connected layer.
 */
public class FullyConnectedLayer: ForwardBackwardLayerProtocol, TrainableLayerProtocol {
  public var name : String
  public var output: [Tensor]
  public var gradient: [Tensor]
  public var weights : Tensor
  public var bias: Tensor
  public var isCpu : Bool
  var parameters : FullyConnectedParameters

  public init(name: String = "fc", parameters: FullyConnectedParameters) {
    self.name = name
    self.parameters = parameters
    self.isCpu = parameters.isCpu
    self.output = []
    self.gradient = []
    // TODO Tensor(dimensions: [parameters.numNeurons, parameters])
    self.weights = Tensor(dimensions: [])
    self.bias = Tensor(dimensions: [])
  }

  func forwardCPU(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0) {
      let bottom = bottomOpt!
      let channels = bottom[0].dimensions[0]
      let height = bottom[0].dimensions[1]
      let width = bottom[0].dimensions[2]
      for i in 0..<parameters.numNeurons {
        output[i].reset(0)
      }
      for i in 0..<bottom.count {
        for j in 0..<parameters.numNeurons {
          for k in 0..<channels {
            for l in 0..<height {
              for m in 0..<width {
                output[i][j] += bottom[i][k, l, m] * weights[k, l, m] + bias[k, l, m]
              }
            }
          }
        }
      }
    }
  }

  func forwardGPU(bottom: [Tensor]?) {
    forwardCPU(bottom)
  }

  func reshape(bottomDimensions: [Int]?) {

  }

  func backwardCPU(top: [Tensor]?) {}
  func backwardGPU(top: [Tensor]?) { backwardCPU(top) }

  func initWeights() {}
  func updateWeights(weightGrad: Tensor) {}


}

public struct FullyConnectedParameters : LayerParameterProtocol {
  public let numNeurons : Int
  public let isBiasTerm : Bool
  public let biasLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightFiller : WeightFiller
  public let biasFiller : WeightFiller
  public let isCpu: Bool
  public init(numNeurons: Int,
              isBiasTerm: Bool = true,
              biasLRMultiplier : Tensor.DataType = 1,
              weightLRMultiplier : Tensor.DataType = 1,
              weightFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              isCpu: Bool) {
    self.numNeurons = numNeurons
    self.isBiasTerm = isBiasTerm
    self.weightFiller = weightFiller
    self.biasFiller = biasFiller
    self.biasLRMultiplier = biasLRMultiplier
    self.weightLRMultiplier = weightLRMultiplier
    self.isCpu = isCpu
  }
}