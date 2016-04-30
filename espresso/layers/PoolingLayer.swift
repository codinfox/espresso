//
//  PollingLayer.swift
//  espresso
//
//  Created by Zhihao Li on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Polling layer.
 */
public class PoolingLayer: ForwardLayerProtocol, BackwardLayerProtocol {
  public var name : String {
    return parameters.name
  }

  public var dependencies: [String] {
    return self.parameters.dependencies
  }

  public var output: Tensor = Tensor()
  public var gradient: Tensor = Tensor()
  var parameters : PoolingParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  public init(parameters: PoolingParameters) {
    self.parameters = parameters
  }

  func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                         bottomDimensions: [[Int]]) {
    switch engine {
    case .CPU:
      self.forwardMethod = forwardCPU
    case .GPU:
      self.forwardMethod = forwardGPU
    }

    self.reshapeByBottomDimensions(bottomDimensions) // may exception (should not)
  }

  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
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
    //    self.gradient.reshape([batchSize, channels, height, width])
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
                      if bottom[currentBatch, currentChannel, row - padSize, col - padSize] > pooled {
                        pooled = bottom[currentBatch, currentChannel, row - padSize, col - padSize]
                      }
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
                      pooled = bottom[currentBatch, currentChannel, row - padSize, col - padSize] / (Tensor.DataType(kernelSize * kernelSize))
                    }
                  }
                }
                output[currentBatch, currentChannel, kernelPositionY / stride, kernelPositionX / stride] += pooled
              }
            }
          }
        }
      }
    }
  }

  func forwardGPU(bottom: [Tensor]) {
    forwardCPU(bottom)
  }
}

public struct PoolingParameters: LayerParameterProtocol {

  public enum PoolingMethod {
    case MAX
    case AVG
  }

  public let name : String
  public let dependencies: [String]
  public let kernelSize : Int
  public let stride : Int
  public let padSize : Int
  public let method : PoolingMethod
  public init(name: String,
              dependencies: [String],
              kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
              method: PoolingMethod = .MAX) {
    self.name = name
    self.dependencies = dependencies
    self.kernelSize = kernelSize
    self.stride = stride
    self.padSize = padSize
    self.method = method
  }
}