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
public class ConvolutionLayer: ForwardLayerProtocol, BackwardLayerProtocol, TrainableLayerProtocol {
  public var name : String
  public var output : [Tensor]
  public var gradient : [Tensor]
  public var weights: Tensor
  public var isCpu : Bool
  var parameters: ConvolutionParameters

  public init(name : String = "conv", parameters: ConvolutionParameters) {
    self.name = name
    self.parameters = parameters
    self.weights = Tensor(dimensions: [parameters.numKerns, parameters.kernelChans, parameters.kernelSize, parameters.kernelSize])
    self.output = []
    self.gradient = [] // Not initialized, needs to be resized
    self.isCpu = parameters.isCpu
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
      let channels = bottom[0].dimensions[1]
      let height = bottom[0].dimensions[2]
      let width = bottom[0].dimensions[3]
      let outHeight = (height + 2 * parameters.padSize - parameters.kernelSize) / parameters.stride + 1
      let outWidth = (width + 2 * parameters.padSize - parameters.kernelSize) / parameters.stride + 1
      for i in 0..<parameters.numKerns {
        output[i].reset(0)
      }
      for i in 0..<bottom.count {
        for kern in 0..<parameters.numKerns {
          for j in 0..<channels {
            for k in 0..<outHeight {
              for l in 0..<outWidth {
                var conved:Float = 0
                for x in 0..<parameters.kernelSize {
                  for y in 0..<parameters.kernelSize {
                    //conved += bottom[i][j, k * parameters.stride + x, l * parameters.stride + y] * weights[kern][x,y]
                    // not considering padSize
                    conved += Float(bottom[i][j, k * parameters.stride + x, l * parameters.stride + y] * weights[kern, j, x, y])
                  }
                }
                output[kern][k, l] += conved
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
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              isCpu: Bool = true) {
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
    self.isCpu = isCpu
  }
}