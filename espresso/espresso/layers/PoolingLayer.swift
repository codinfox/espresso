//
//  PollingLayer.swift
//  espresso
//
//  Created by Zhihao Li on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief Polling layer.
 */
public class PoolingLayer: ForwardLayerProtocol, BackwardLayerProtocol {
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
  var parameters : PoolingParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  public init(parameters: PoolingParameters) {
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
    if self.parameters.globalPooling {
      let oneBottomDimensionsSample = bottomDimensions[0]
      let bottomHeight = oneBottomDimensionsSample[2]

      // TODO: should support height != width
      self.parameters.kernelSize = bottomHeight
    }
    self.output = Tensor(metalDevice: metalDevice)
    self.gradient = Tensor(metalDevice: metalDevice)
    self.reshapeByBottomDimensions(bottomDimensions) // may exception (should not)
  }

  public func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
    let oneBottomDimensionsSample = bottomDimensions[0]
    // subject to change, currently just 4 dimensions
    let batchSize = oneBottomDimensionsSample[0]
    let bottomChannels = oneBottomDimensionsSample[1]
    let bottomHeight = oneBottomDimensionsSample[2]
    let bottomWidth = oneBottomDimensionsSample[3]

    let channels = bottomChannels
    let height = (bottomHeight + self.parameters.padSize * 2 - self.parameters.kernelSize) / self.parameters.stride + 1
    let width = (bottomWidth + self.parameters.padSize * 2 - self.parameters.kernelSize) / self.parameters.stride + 1

    self.output.reshape([batchSize, channels, height, width])
    //self.gradient.reshape([batchSize, channels, height, width])
  }

  public func forwardCPU(bottom: [Tensor]) {
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

      let paddedHeight = bottomHeight + 2 * padSize
      let paddedWidth = bottomWidth + 2 * padSize
      let boundKernelPositionY = (paddedHeight - kernelSize + stride) / stride * stride
      let boundKernelPositionX = (paddedWidth - kernelSize + stride) / stride * stride

      output.reset(0)

      for currentBatch in 0 ..< batchSize {
        for currentChannel in 0 ..< bottomChannels {
          for kernelPositionY in 0.stride(to: boundKernelPositionY, by: stride) {
            for kernelPositionX in 0.stride(to: boundKernelPositionX, by: stride) {
              var pooled: Tensor.DataType
              switch parameters.method {
              case .MAX:
                pooled = -Tensor.DataType.infinity
                for y in 0 ..< kernelSize {
                  for x in 0 ..< kernelSize {
                    let row = kernelPositionY + y
                    let col = kernelPositionX + x
                    if row >= padSize && row < paddedHeight - padSize && col >= padSize && col < paddedWidth - padSize {
                      pooled = max(pooled, bottom[currentBatch, currentChannel, row - padSize, col - padSize])
                    } else {
                      if 0 > pooled {
                        pooled = 0
                      }
                    }
                  }
                }
              case .AVG:
                pooled = 0
                for y in 0 ..< kernelSize {
                  for x in 0 ..< kernelSize {
                    let row = kernelPositionY + y
                    let col = kernelPositionX + x
                    if row >= padSize && row < paddedHeight - padSize && col >= padSize && col < paddedWidth - padSize {
                      pooled += bottom[currentBatch, currentChannel, row - padSize, col - padSize] / (Tensor.DataType(kernelSize * kernelSize))
                    }
                  }
                }
              }
              output[currentBatch, currentChannel, kernelPositionY / stride, kernelPositionX / stride] = pooled
            }
          }
        }
      }
    }
  }

  public func forwardGPU(bottom: [Tensor]) {
    if (bottom.count > 0) {
      let bottom = bottom[0]
      let commandBuffer = self.metalCommandQueue.commandBuffer()

      let padSize = Int32(parameters.padSize)
      let kernelSize = Int32(parameters.kernelSize)
      let stride = Int32(parameters.stride)

      let batchSize = Int32(bottom.dimensions[0])
      let inputChannel = Int32(bottom.dimensions[1])
      let inputHeight = Int32(bottom.dimensions[2])
      let inputWidth = Int32(bottom.dimensions[3])

      let outputChannel = inputChannel
      let outputHeight = Int32((inputHeight + 2 * padSize - kernelSize) / stride + 1)
      let outputWidth = Int32((inputWidth + 2 * padSize - kernelSize) / stride + 1)

      let count = batchSize * outputChannel * outputHeight * outputWidth

      // copy the parameters to metal
      let paramBuffer = createPoolingParameter(MetalPoolingParameter(count: count, padSize: padSize, kernelSize: kernelSize, stride: stride, inputChannel: inputChannel, inputHeight: inputHeight, inputWidth: inputWidth, outputChannel: outputChannel, outputHeight: outputHeight, outputWidth: outputWidth), metalDevice: self.metalDevice)
      // perform computation
      var funcName = ""
      if (parameters.method == .MAX) {
        funcName = "poolingMaxForward"
      } else {
        funcName = "poolingAvgForward"
      }
      submitCommonComputeJob(funcName, paramBuffer: paramBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice, inputData: bottom, outputData: self.output, commandBuffer: commandBuffer, threadCount: Int(count))
      self.output.getFromMetal()
    }
  }
}

public struct PoolingParameters: LayerParameterProtocol {

  public enum PoolingMethod {
    case MAX
    case AVG
  }

  public let name : String
  public let dependencies: [String]
  public var kernelSize : Int // TODO: bad?
  public let stride : Int
  public let padSize : Int
  public let method : PoolingMethod
  public let globalPooling : Bool
  public init(name: String,
              dependencies: [String],
              kernelSize: Int = 1,
              stride: Int = 1,
              padSize: Int = 0,
              method: PoolingMethod = .MAX,
              globalPooling : Bool = false) {
    self.name = name
    self.dependencies = dependencies
    self.kernelSize = kernelSize
    self.stride = stride
    self.padSize = padSize
    self.method = method
    self.globalPooling = globalPooling
  }
}
