//
//  ConvolutionLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal
import Accelerate

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

  public var output: Tensor!
  public var gradient: Tensor!
  public var weights: Tensor!
  public var bias: Tensor!
  var parameters: ConvolutionParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  var compressedInfo: CompressedInfo! = nil

  public init(parameters: ConvolutionParameters) {
    self.parameters = parameters
  }

  public func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                                bottomDimensions: [[Int]],
                                metalDevice: MTLDevice! = nil,
                                metalDefaultLibrary: MTLLibrary! = nil,
                                metalCommandQueue: MTLCommandQueue! = nil) {
    switch engine {
    case .CPU:
      self.forwardMethod = forwardCPU
    case .GPU:
      self.forwardMethod = forwardGPU
    }
    self.metalDevice = metalDevice
    self.metalDefaultLibrary = metalDefaultLibrary
    self.metalCommandQueue = metalCommandQueue
    self.output = Tensor(metalDevice: metalDevice)
    self.gradient = Tensor(metalDevice: metalDevice)
    self.weights = Tensor(metalDevice: metalDevice)
    self.bias = Tensor(metalDevice: metalDevice)
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
    self.bias.reshape([channels])
    self.output.reshape([batchSize, channels, height, width])
    self.gradient.reshape([batchSize, channels, height, width])
  }

  func forwardCPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      if self.memoryLimitedMode {
        restoreWeightsByDecompression()
      }
      let bottom = bottom[0]
      let batchSize = bottom.dimensions[0]
      let bottomChannels = bottom.dimensions[1]
      let bottomHeight = bottom.dimensions[2]
      let bottomWidth = bottom.dimensions[3]

      let padSize = parameters.padSize
      let stride = parameters.stride
      let kernelSize = parameters.kernelSize
      let numOutput = parameters.numOutput

      let weightElementsOneChannelAllOutputs = numOutput * kernelSize * kernelSize

      let outputHeight = (bottomHeight + 2 * padSize - kernelSize) / stride + 1
      let outputWidth = (bottomWidth + 2 * padSize - kernelSize) / stride + 1

      // FIXME: Only support batch size 1
      var mulRes = [Float](count : numOutput * outputHeight * outputWidth, repeatedValue : 0.0)

      if self.memoryLimitedMode {
        // if memory limited
        // FIXME: No impact on memory usage
        var weightTrans = [Float](count: self.weights.storage.count, repeatedValue: 0.0)
        vDSP_mtrans(self.weights.storage, 1, &weightTrans, 1, UInt(bottomChannels * kernelSize * kernelSize), UInt(numOutput))
        var tmpMulRes = [Float](count : numOutput * outputHeight * outputWidth, repeatedValue : 0.0)

        for currentBatch in 0 ..< batchSize {
          for currentChannel in 0 ..< bottomChannels {
            let inputOffset = currentBatch * bottomChannels * bottomHeight * bottomWidth + currentChannel * bottomHeight * bottomWidth 

            let bottomCol = im2colCpu(bottom.storage, inputChannels: 1, height: bottomHeight, width: bottomWidth, kernelSize: kernelSize, padSize: padSize, stride: stride, inputOffset: inputOffset)
            var bottomColTrans = [Float](count: bottomCol.count, repeatedValue: 0.0)
            vDSP_mtrans(bottomCol, 1, &bottomColTrans, 1, UInt(outputHeight * outputWidth), UInt(kernelSize * kernelSize))

            let weightSlice : [Float] = Array(weightTrans[weightElementsOneChannelAllOutputs * currentChannel ..< weightElementsOneChannelAllOutputs * (currentChannel + 1)])

            vDSP_mmul(bottomColTrans, 1, weightSlice, 1, &tmpMulRes, 1, UInt(outputHeight * outputWidth), UInt(numOutput), UInt(kernelSize * kernelSize))
            vDSP_vadd(mulRes, 1, tmpMulRes, 1, &mulRes, 1, vDSP_Length(mulRes.count))
          }
        }
        vDSP_mtrans(mulRes, 1, &mulRes, 1, UInt(numOutput), UInt(outputWidth * outputHeight))

      } else {
        // if memory is not an issue
        let bottomCol = im2colCpu(bottom.storage, inputChannels: bottomChannels, height: bottomHeight, width: bottomWidth, kernelSize: kernelSize, padSize: padSize, stride: stride)

        let weightCol = self.weights.storage

        vDSP_mmul(weightCol, 1, bottomCol, 1, &mulRes, 1, UInt(numOutput), UInt(outputHeight * outputWidth), UInt(bottomChannels * kernelSize * kernelSize))
      }

      let biasCol:[Float] = (0..<mulRes.count).map({self.bias.storage[Int($0 / (outputHeight * outputWidth))]})
      vDSP_vadd(mulRes, 1, biasCol, 1, &self.output.storage, 1, vDSP_Length(mulRes.count))

      if self.memoryLimitedMode {
        self.weights.purgeStorage()
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
      let (computeCommandEncoder, computePipelineState) = setupComputEncoder("convolutionForward", commandBuffer: commandBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice)
      computeCommandEncoder.setBuffer(bottom.mtlStorage, offset: 0, atIndex: 0)
      computeCommandEncoder.setBuffer(self.output.mtlStorage, offset: 0, atIndex: 1)
      computeCommandEncoder.setBuffer(self.weights.mtlStorage, offset: 0, atIndex: 2)
      computeCommandEncoder.setBuffer(self.bias.mtlStorage, offset: 0, atIndex: 3)
      computeCommandEncoder.setBuffer(paramBuffer, offset: 0, atIndex: 4)
      submitComputeJob(computeCommandEncoder, computePipelineState: computePipelineState, count: 0)
      commandBuffer.commit()
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
