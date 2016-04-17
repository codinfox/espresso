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
  public var output : [Tensor]
  public var gradient : [Tensor]
  public var weights: Tensor
  public var bias : Tensor
  public var engine: NetworkProperties.NetworkEngine
  var parameters: ConvolutionParameters

  public init(name : String = "conv", parameters: ConvolutionParameters) {
    self.name = name
    self.parameters = parameters
    self.weights = Tensor(dimensions: [parameters.numKerns, parameters.kernelChans, parameters.kernelSize, parameters.kernelSize])
    self.output = []
    self.gradient = [] // Not initialized, needs to be resized
    self.engine = .CPU
  }

  public func reshape(bottomDimensions: [Int]?) {
    // Resize output and gradient
  }

  public func backward(top: [Tensor]?) {

  }

  public func initWeights() {
    // weightFiller, biasFiller --> self.weights
  }

  public func updateWeights(weightGrad: Tensor) {

  }

  func forward_cpu(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0){
      let bottom = bottomOpt!
      let batchSize = bottom.count
      let channels = bottom[0].dimensions[0]
      let height = bottom[0].dimensions[1]
      let width = bottom[0].dimensions[2]
      let padSize = parameters.padSize
      let stride = parameters.stride
      let kernelSize = parameters.kernelSize
      let padedHeight = (height + 2 * padSize - kernelSize + 1)
      let padedWidth = (width + 2 * padSize - kernelSize + 1)
      for i in 0..<batchSize {
        output[i].reset(0)
      }
      for i in 0..<batchSize {
        for kern in 0..<parameters.numKerns {
          for j in 0..<channels {
            for k in 0.stride(to: padedHeight, by: stride) {
              for l in 0.stride(to: padedWidth, by: stride) {
                var conved:Float = 0
                for x in 0..<parameters.kernelSize {
                  for y in 0..<parameters.kernelSize {
                    let row = k + x
                    let col = l + y
                    if (row >= padSize && row < padedHeight - padSize
                        && col >= padSize && col < padedWidth) {
                      conved += bottom[i][j, row, col] * weights[kern, j, x, y]
                    }
                  }
                }
                output[i][kern, k, l] += conved
              }
            }
          }
        }
      }
    }
  }

  func forward_gpu(bottom: [Tensor]?) {
    forward_cpu(bottom)
  }

  func backward_cpu(top: [Tensor]?) {

  }

  func backward_gpu(top: [Tensor]?) {
    backward_cpu(top)
  }

  public func layerSetUp(networkProperties: NetworkProperties) {
  }
}

public struct ConvolutionParameters: LayerParameterProtocol {
  public let numKerns : Int
  public let kernelChans: Int
  public let kernelSize : Int
  public let stride : Int
  public let padSize : Int
  public let isBiasTerm : Bool
  public let biasLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightFiller : WeightFiller
  public let biasFiller : WeightFiller
  public let isCpu : Bool
  public init(numKerns: Int,
              kernelChans: Int,
              kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
              isBiasTerm: Bool = true,
              biasLRMultiplier : Tensor.DataType = 1,
              weightLRMultiplier : Tensor.DataType = 1,
              weightFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1)) {
    self.numKerns = numKerns
    self.kernelChans = kernelChans
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