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
  public var output: [Tensor] = []
  public var gradient: [Tensor] = []
  public var weights : Tensor = Tensor()
  public var bias: Tensor = Tensor()
  public var engine: NetworkProperties.NetworkEngine = .CPU
  var parameters : FullyConnectedParameters

  public init(name: String = "fc", parameters: FullyConnectedParameters) {
    self.name = name
    self.parameters = parameters
  }

  func layerSetUp(networkProperties: NetworkProperties, bottomNumOutput: Int? = nil) {
    guard bottomNumOutput != nil && bottomNumOutput > 0 else {
      // TODO: throw exception
      return
    }
    // Fully connected layer will not initialize weights in the layerSetUp as it needs the output dimension of the bottom layer
    self.engine = networkProperties.engine
    if (self.parameters.isBiasTerm) {
      self.bias.reshape([self.parameters.numOutput])
    }

    // Set batch size
    for _ in 0 ..< networkProperties.batchSize {
      self.output.append(Tensor(dimensions: [self.parameters.numOutput]))
      self.gradient.append(Tensor(dimensions: [self.parameters.numOutput]))
    }
  }

  func numOutput() -> Int {
    return parameters.numOutput
  }

  func reshape(bottomDimensionsOpt: [Int]?) {
    if let bottomDimensionsOpt = bottomDimensionsOpt {
      var bottomDimensions = bottomDimensionsOpt
      bottomDimensions.insert(parameters.numOutput, atIndex: 0)
      self.weights.reshape(bottomDimensions)
    }
  }

  func forwardCPU(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0){
      let bottom = bottomOpt!
      let batchSize = bottom.count

      let numOutput = parameters.numOutput

      for currentBatch in 0 ..< batchSize {
        for currentOutput in 0 ..< numOutput {
          var tmpResult : Tensor.DataType = 0
          // FIXME: bad API design
          for i in bottom[currentBatch].storage.indices {
            // FIXME: Hack
            tmpResult += self.weights[currentOutput, 0, 0, i] * bottom[currentBatch].storage[i]
          }
          self.output[currentBatch][currentOutput] = tmpResult
        }
      }
    }
  }

  func forwardGPU(bottom: [Tensor]?) {
    forwardCPU(bottom)
  }

  func backwardCPU(top: [Tensor]?) {}
  func backwardGPU(top: [Tensor]?) { backwardCPU(top) }

  func initWeights() {}
  func updateWeights(weightGrad: Tensor) {}


}

public struct FullyConnectedParameters : LayerParameterProtocol {
  public let numOutput : Int
  public let isBiasTerm : Bool
  public let biasLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightFiller : WeightFiller
  public let biasFiller : WeightFiller
  public init(numOutput: Int,
              isBiasTerm: Bool = true,
              biasLRMultiplier : Tensor.DataType = 1,
              weightLRMultiplier : Tensor.DataType = 1,
              weightFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1)) {
    self.numOutput = numOutput
    self.isBiasTerm = isBiasTerm
    self.weightFiller = weightFiller
    self.biasFiller = biasFiller
    self.biasLRMultiplier = biasLRMultiplier
    self.weightLRMultiplier = weightLRMultiplier
  }
}