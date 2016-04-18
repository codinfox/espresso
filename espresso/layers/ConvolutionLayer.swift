//
//  ConvolutionLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Convolution layer.
 */
public class ConvolutionLayer: ForwardBackwardLayerProtocol, TrainableLayerProtocol {
  public var name : String
  public var engine: NetworkProperties.NetworkEngine = .CPU
  public var output : [Tensor] = []
  public var gradient : [Tensor] = []
  public var weights: Tensor = Tensor()
  public var bias: Tensor = Tensor()
  var parameters: ConvolutionParameters

  public init(name : String = "conv", parameters: ConvolutionParameters) {
    self.name = name
    self.parameters = parameters
  }

  func layerSetUp(networkProperties: NetworkProperties, bottomNumOutput: Int? = nil) {
    guard bottomNumOutput != nil && bottomNumOutput > 0 else {
      // TODO: throw exception
      return
    }

    self.weights.reshape([self.parameters.numOutput, bottomNumOutput!, self.parameters.kernelSize, self.parameters.kernelSize]);
    self.engine = networkProperties.engine
    if (self.parameters.isBiasTerm) {
      self.bias.reshape([self.parameters.numOutput])
    }

    // Set batch size
    for _ in 0 ..< networkProperties.batchSize {
      self.output.append(Tensor())
      self.gradient.append(Tensor())
    }
  }

  func numOutput() -> Int {
    return parameters.numOutput
  }

  public func backward(top: [Tensor]?) {

  }

  public func initWeights() {
    // weightFiller, biasFiller --> self.weights
  }

  public func updateWeights(weightGrad: Tensor) {

  }

  func reshape(bottomDimensionsOpt: [Int]?) {
    if let bottomDimensionsOpt = bottomDimensionsOpt {

      // subject to change, currently just 3 dimensions
      let bottomHeight = bottomDimensionsOpt[1]
      let bottomWidth = bottomDimensionsOpt[2]

      let channels = self.parameters.numOutput
      let height = (bottomHeight + self.parameters.padSize * 2 - self.parameters.kernelSize + 1) / self.parameters.stride
      let width = (bottomWidth + self.parameters.padSize * 2 - self.parameters.kernelSize + 1) / self.parameters.stride

      for i in self.output.indices {
        self.output[i].reshape([channels, height, width])
        self.gradient[i].reshape([channels, height, width])
      }
    }
  }

  func forwardCPU(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0){
      let bottom = bottomOpt!
      let batchSize = bottom.count

      let bottomChannels = bottom[0].dimensions[0]
      let bottomHeight = bottom[0].dimensions[1]
      let bottomWidth = bottom[0].dimensions[2]

      let padSize = parameters.padSize
      let stride = parameters.stride
      let kernelSize = parameters.kernelSize
      let numOutput = parameters.numOutput

      let padedHeight = (bottomHeight + 2 * padSize - kernelSize + 1)
      let padedWidth = (bottomWidth + 2 * padSize - kernelSize + 1)

      for i in 0 ..< batchSize {
        output[i].reset(0)
      }

      for currentBatch in 0 ..< batchSize {
        for currentKernel in 0 ..< numOutput {
          for currentChannel in 0 ..< bottomChannels {
            for kernelPositionY in 0.stride(to: padedHeight, by: stride) {
              for kernelPositionX in 0.stride(to: padedWidth, by: stride) {
                var conved: Tensor.DataType = 0
                for y in 0 ..< kernelSize {
                  for x in 0 ..< kernelSize {
                    let row = kernelPositionY + y
                    let col = kernelPositionX + x
                    if row >= padSize && row < padedHeight - padSize && col >= padSize && col < padedWidth - padSize {
                      conved += bottom[currentBatch][currentChannel, row - padSize, col - padSize] * weights[currentKernel, currentChannel, y, x]
                    }
                  }
                }
                output[currentBatch][currentKernel, kernelPositionY / stride, kernelPositionX / stride] += conved
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

  func backwardCPU(top: [Tensor]?) {

  }

  func backwardGPU(top: [Tensor]?) {
    backwardCPU(top)
  }

}

public struct ConvolutionParameters: LayerParameterProtocol {
  public let numOutput : Int
  public let kernelSize : Int
  public let stride : Int
  public let padSize : Int
  public let isBiasTerm : Bool
  public let biasLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightFiller : WeightFiller
  public let biasFiller : WeightFiller
  public init(numOutput: Int,
              kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
              isBiasTerm: Bool = true,
              biasLRMultiplier : Tensor.DataType = 1,
              weightLRMultiplier : Tensor.DataType = 1,
              weightFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1)) {
    self.numOutput = numOutput
    self.kernelSize = kernelSize
    self.stride = stride
    self.padSize = padSize
    self.isBiasTerm = isBiasTerm
    self.weightFiller = weightFiller
    self.biasFiller = biasFiller
    self.biasLRMultiplier = biasLRMultiplier
    self.weightLRMultiplier = weightLRMultiplier
  }
}