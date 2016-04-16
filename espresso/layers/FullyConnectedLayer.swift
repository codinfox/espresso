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
  public var isCpu : Bool
  var parameters : FullyConnectedParameters

  public init(name: String = "fc", parameters: FullyConnectedParameters) {
    self.name = name
    self.parameters = parameters
    self.isCpu = parameters.isCpu
  }

  func forward_cpu(bottom: [Tensor]?) {

  }
  func forward_gpu(bottom: [Tensor]?) {
    forward_cpu(bottom)
  }

  func reshape(bottomDimensions: [Int]?) {

  }

  func backward_cpu(top: [Tensor]?) {}
  func backward_gpu(top: [Tensor]?) { backward_cpu(top) }

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
              kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
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