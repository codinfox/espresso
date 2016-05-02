//
//  ConvolutionLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief Convolution layer.
 */
public class ConvolutionLayer: ForwardLayerProtocol, BackwardLayerProtocol, TrainableLayerProtocol {
  public var name : String {
    return parameters.name
  }

  public var dependencies: [String] {
    return self.parameters.dependencies
  }

  public var metalDevice: MTLDevice!
  public var metalCommandQueue: MTLCommandQueue!
  public var metalDefaultLibrary: MTLLibrary!

  public var output: Tensor = Tensor()
  public var gradient: Tensor = Tensor()
  public var weights: Tensor = Tensor()
  public var bias: Tensor = Tensor()
  var parameters: ConvolutionParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  public init(parameters: ConvolutionParameters) {
    self.parameters = parameters
  }

  public func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                                bottomDimensions: [[Int]],
                                metalDevice: MTLDevice!,
                                metalDefaultLibrary: MTLLibrary!,
                                metalCommandQueue: MTLCommandQueue!) {
    switch engine {
    case .CPU:
      self.forwardMethod = forwardCPU
    case .GPU:
      self.forwardMethod = forwardGPU
    }
    self.metalDevice = metalDevice
    self.metalDefaultLibrary = metalDefaultLibrary
    self.metalCommandQueue = metalCommandQueue
    self.reshapeByBottomDimensions(bottomDimensions) // may exception (should not)
  }

  func outputDimensions() -> [[Int]] {
    return [output.dimensions]
  }

  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
    let oneBottomDimensionsSample = bottomDimensions[0]
    // subject to change, currently just 4 dimensions
    let batchSize = oneBottomDimensionsSample[0]
    let bottomChannels = oneBottomDimensionsSample[1]
    let bottomHeight = oneBottomDimensionsSample[2]
    let bottomWidth = oneBottomDimensionsSample[3]

    let channels = self.parameters.numOutput
    let height = (bottomHeight + self.parameters.padSize * 2 - self.parameters.kernelSize) / self.parameters.stride + 1
    let width = (bottomWidth + self.parameters.padSize * 2 - self.parameters.kernelSize) / self.parameters.stride + 1

    self.weights.reshape([channels, bottomChannels, self.parameters.kernelSize, self.parameters.kernelSize])
    if self.parameters.isBiasTerm {
      self.bias.reshape([channels])
    }
    self.output.reshape([batchSize, channels, height, width])
    self.gradient.reshape([batchSize, channels, height, width])
  }

  func forwardCPU(bottom: [Tensor]) {
    // Preprocess bottom to fit this layer
    if bottom.count > 0 {
      let bottom = bottom[0] // in conv layer, bottom is really just a single Tensor

      let batchSize = bottom.dimensions[0]
      let bottomChannels = bottom.dimensions[1]
      let bottomHeight = bottom.dimensions[2]
      let bottomWidth = bottom.dimensions[3]

      let padSize = parameters.padSize
      let stride = parameters.stride
      let kernelSize = parameters.kernelSize
      let numOutput = parameters.numOutput

      let paddedHeight = bottomHeight + 2 * padSize
      let paddedWidth = bottomWidth + 2 * padSize
      let boundKernelPositionY = (paddedHeight - kernelSize + stride) / stride * stride
      let boundKernelPositionX = (paddedWidth - kernelSize + stride) / stride * stride

      output.reset(0)

      for currentBatch in 0 ..< batchSize {
        for currentKernel in 0 ..< numOutput {
          for currentChannel in 0 ..< bottomChannels {
            for kernelPositionY in 0.stride(to: boundKernelPositionY, by: stride) {
              for kernelPositionX in 0.stride(to: boundKernelPositionX, by: stride) {
                var conved: Tensor.DataType = 0
                for y in 0 ..< kernelSize {
                  for x in 0 ..< kernelSize {
                    let row = kernelPositionY + y
                    let col = kernelPositionX + x
                    if row >= padSize && row < paddedHeight - padSize && col >= padSize && col < paddedWidth - padSize {
                      conved += bottom[currentBatch, currentChannel, row - padSize, col - padSize] * weights[currentKernel, currentChannel, y, x]
                    }
                  }
                }
                output[currentBatch, currentKernel, kernelPositionY / stride, kernelPositionX / stride] += conved
                if currentChannel == 0 && parameters.isBiasTerm {
                  output[currentBatch, currentKernel, kernelPositionY / stride, kernelPositionX / stride] += bias[currentKernel]
                }
              }
            }
          }
        }
      }
    }
  }

  func forwardGPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0]
      let commandBuffer = self.metalCommandQueue.commandBuffer()

      let padSize = parameters.padSize
      let kernelSize = parameters.kernelSize
      let stride = parameters.stride

      let inputChannel = bottom.dimensions[1]
      let inputHeight = bottom.dimensions[2]
      let inputWidth = bottom.dimensions[3]

      let outputChannel = parameters.numOutput
      let outputHeight = (inputHeight + 2 * padSize - kernelSize) / stride
      let outputWidth = (inputWidth + 2 * padSize - kernelSize) / stride

      // copy the parameters to metal
      let paramBuffer = createConvolutionParameter(MetalConvolutionParameter(padSize: padSize, kernelSize: kernelSize, stride: stride, inputChannel: inputChannel, inputHeight: inputHeight, inputWidth: inputWidth, outputChannel: outputChannel, outputHeight: outputHeight, outputWidth: outputWidth), metalDevice: self.metalDevice)
      // perform computation
      submitComputeJob("convolutionForward", paramBuffer: paramBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice, inputData: bottom, outputData: self.output, commandBuffer: commandBuffer)
      commandBuffer.waitUntilCompleted()
    }
  }
}

public struct ConvolutionParameters: LayerParameterProtocol {
  public let name : String
  public let dependencies: [String]
  public let numOutput : Int
  public let kernelSize : Int
  public let stride : Int
  public let padSize : Int
  public let isBiasTerm : Bool
  public let biasLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightLRMultiplier : Tensor.DataType // learning rate multiplier
  public let weightFiller : WeightFiller
  public let biasFiller : WeightFiller
  public init(name: String,
              dependencies: [String],
              numOutput: Int,
              kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
              isBiasTerm: Bool = true,
              biasLRMultiplier : Tensor.DataType = 1,
              weightLRMultiplier : Tensor.DataType = 1,
              weightFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1),
              biasFiller: WeightFiller = gaussianWeightFiller(mean: 0, std: 1)) {
    self.name = name
    self.dependencies = dependencies
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
