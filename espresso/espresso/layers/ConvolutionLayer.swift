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

  public func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
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

  public func forwardCPU(bottom: [Tensor]) {
    if bottom.count > 0 {
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
      let bottomCol = im2colCpu(bottom.storage, inputChannels: bottomChannels, height: bottomHeight, width: bottomWidth, kernelSize: kernelSize, padSize: padSize, stride: stride)

      if self.memoryLimitedMode {
        // if memory limited
        mulRes = sparseDenseMatrixMultiplication(self.compressedInfo, dense: bottomCol, M: (numOutput), N: (outputHeight * outputWidth), P: (bottomChannels * kernelSize * kernelSize))
      } else {
        // if memory is not an issue

        let weightCol = self.weights.storage

        vDSP_mmul(weightCol, 1, bottomCol, 1, &mulRes, 1, UInt(numOutput), UInt(outputHeight * outputWidth), UInt(bottomChannels * kernelSize * kernelSize))
      }


      let biasCol:[Float] = (0..<mulRes.count).map({self.bias.storage[Int($0 / (outputHeight * outputWidth))]})
      vDSP_vadd(mulRes, 1, biasCol, 1, &self.output.storage, 1, vDSP_Length(mulRes.count))
    }
  }

  //for debugging
  public func forwardGPUDebug(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0]
      bottom.getFromMetal()
      self.output.storage = [Float](count: self.output.count(), repeatedValue: 0)
      self.forwardCPU([bottom])
      self.output.putToMetal()
    }
  }

  public func forwardGPUOptAttempt(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0]

      let padSize = parameters.padSize
      let kernelSize = parameters.kernelSize
      let stride = parameters.stride

      let batchSize = bottom.dimensions[0]
      let inputChannel = bottom.dimensions[1]
      let inputHeight = bottom.dimensions[2]
      let inputWidth = bottom.dimensions[3]

      let outputChannel = parameters.numOutput
      let outputHeight = (inputHeight + 2 * padSize - kernelSize) / stride + 1
      let outputWidth = (inputWidth + 2 * padSize - kernelSize) / stride + 1

      // set output to bias
      self.output.storage = (0..<self.output.count()).map({self.bias.storage[Int($0 / (outputHeight * outputWidth))]})
      self.output.putToMetal()

      let colDimension = [1, 1, inputChannel * kernelSize * kernelSize, outputHeight * outputWidth];
      let count = batchSize * outputChannel * outputHeight * outputWidth
      let bottomCol = Tensor(metalDevice: metalDevice)
      bottomCol.reshape(colDimension)
      im2colGpu(bottom, output: bottomCol, inputChannels: inputChannel, height: inputHeight, width: inputWidth, kernelSize: kernelSize, padSize: padSize, stride: stride, metalDevice: metalDevice, metalCommandQueue: metalCommandQueue, metalDefaultLibrary: metalDefaultLibrary)
      // batchSize
      espressoSgemm(false, transB: false, m: UInt32(outputChannel), n: UInt32(outputHeight * outputWidth), k: UInt32(inputChannel * kernelSize * kernelSize), alpha: 1, A: self.weights, B: bottomCol, beta: 1, C: self.output, metalDevice: self.metalDevice, metalDefaultLibrary: self.metalDefaultLibrary, metalCommandQueue: self.metalCommandQueue)
      //appleMM(UInt16(outputChannel), n: UInt16(inputChannel * kernelSize * kernelSize), k: UInt16(outputHeight * outputWidth), A: self.weights, B: bottomCol, C: self.output, metalDevice: self.metalDevice, metalDefaultLibrary: self.metalDefaultLibrary, metalCommandQueue: self.metalCommandQueue)
    }
  }
  public func forwardGPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0]

      let padSize = Int32(parameters.padSize)
      let kernelSize = Int32(parameters.kernelSize)
      let stride = Int32(parameters.stride)

      let batchSize = Int32(bottom.dimensions[0])
      let inputChannel = Int32(bottom.dimensions[1])
      let inputHeight = Int32(bottom.dimensions[2])
      let inputWidth = Int32(bottom.dimensions[3])

      let outputChannel = Int32(parameters.numOutput)
      let outputHeight = Int32((inputHeight + 2 * padSize - kernelSize) / stride + 1)
      let outputWidth = Int32((inputWidth + 2 * padSize - kernelSize) / stride + 1)

      let count = Int32(batchSize * outputChannel * outputHeight * outputWidth)

      // copy the parameters to metal
      let paramBuffer = createConvolutionParameter(MetalConvolutionParameter(count: count, padSize: padSize, kernelSize: kernelSize, stride: stride, inputChannel: inputChannel, inputHeight: inputHeight, inputWidth: inputWidth, outputChannel: outputChannel, outputHeight: outputHeight, outputWidth: outputWidth), metalDevice: self.metalDevice)

      let commandBuffer = self.metalCommandQueue.commandBuffer()
      // perform computation
      let (computeCommandEncoder, computePipelineState) = setupComputEncoder("convolutionForward", commandBuffer: commandBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice)
      computeCommandEncoder.setBuffer(bottom.mtlStorage, offset: 0, atIndex: 0)
      computeCommandEncoder.setBuffer(self.output.mtlStorage, offset: 0, atIndex: 1)
      computeCommandEncoder.setBuffer(self.weights.mtlStorage, offset: 0, atIndex: 2)
      computeCommandEncoder.setBuffer(self.bias.mtlStorage, offset: 0, atIndex: 3)
      computeCommandEncoder.setBuffer(paramBuffer, offset: 0, atIndex: 4)
      submitComputeJob(computeCommandEncoder, computePipelineState: computePipelineState, count: Int(count))
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
