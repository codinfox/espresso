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
public class FullyConnectedLayer: ForwardLayerProtocol, BackwardLayerProtocol, TrainableLayerProtocol {
  public var name : String {
    return parameters.name
  }

  public var dependencies: [String] {
    return self.parameters.dependencies
  }

  public var output: Tensor = Tensor()
  public var gradient: Tensor = Tensor()
  public var weights: Tensor = Tensor()
  public var bias: Tensor = Tensor()
  var parameters : FullyConnectedParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  public init(parameters: FullyConnectedParameters) {
    self.parameters = parameters
  }

  func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                         bottomDimensions: [[Int]]? = nil) {
    switch engine {
    case .CPU:
      self.forwardMethod = forwardCPU
    case .GPU:
      self.forwardMethod = forwardGPU
    }

    self.reshapeByBottomDimensions(bottomDimensions!) // may exception (should not)
  }

  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
    let oneBottomDimensionsSample = bottomDimensions[0]
    // subject to change, currently just 4 dimensions
    let batchSize = oneBottomDimensionsSample[0]
    let channels = self.parameters.numOutput

    var weightDimensions = oneBottomDimensionsSample
    weightDimensions[0] = channels // change batchSize to channels of current layer, hacky
    self.weights.reshape(weightDimensions)
    if (self.parameters.isBiasTerm) {
      self.bias.reshape([channels])
    }
    self.output.reshape([batchSize, channels])
//    self.gradient.reshape([batchSize, channels])
  }

  func forwardCPU(bottom: [Tensor]?) {
    // Preprocess bottom to fit this layer
    if let bottom = bottom where bottom.count > 0 {
      let bottom = bottom[0] // in fc layer, bottom is really just a single Tensor

      let batchSize = bottom.dimensions[0]
      let numOutput = parameters.numOutput

      let numElementsPerBatch = bottom.count(fromDimension: 1)
      assert(numElementsPerBatch == self.weights.count() / numOutput, "Num elements not match")

      for currentBatch in 0 ..< batchSize {
        for currentOutput in 0 ..< numOutput {
          var tmpResult : Tensor.DataType = 0
          // FIXME: bad API design
          for i in 0 ..< numElementsPerBatch {
            // FIXME: Hack
            tmpResult += self.weights[currentOutput, 0, 0, i] * bottom.storage[currentBatch * numElementsPerBatch + i]
          }
          self.output[currentBatch, currentOutput] = tmpResult
        }
      }
    }
  }

  func forwardGPU(bottom: [Tensor]?) {
    forwardCPU(bottom)
  }
}

public struct FullyConnectedParameters : LayerParameterProtocol {
  public let name : String
  public let dependencies: [String]
  public let numOutput : Int
  public let isBiasTerm : Bool
  public let biasLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightFiller : WeightFiller
  public let biasFiller : WeightFiller
  public init(name: String,
              dependencies: [String],
              numOutput: Int,
              isBiasTerm: Bool = true,
              biasLRMultiplier : Tensor.DataType = 1,
              weightLRMultiplier : Tensor.DataType = 1,
              weightFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1)) {
    self.name = name
    self.dependencies = dependencies
    self.numOutput = numOutput
    self.isBiasTerm = isBiasTerm
    self.weightFiller = weightFiller
    self.biasFiller = biasFiller
    self.biasLRMultiplier = biasLRMultiplier
    self.weightLRMultiplier = weightLRMultiplier
  }
}