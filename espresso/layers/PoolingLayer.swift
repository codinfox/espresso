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
public class PoolingLayer: ForwardBackwardLayerProtocol {
  public var name : String
  var parameters : PoolingParameters
  public var output: [Tensor] = []
  public var gradient: [Tensor] = []
  public var engine: NetworkProperties.NetworkEngine = .CPU
  private var myNumOutput : Int = 0

  public init(name: String = "pooling", parameters: PoolingParameters) {
    self.name = name
    self.parameters = parameters
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

      let paddedHeight = bottomHeight + 2 * padSize
      let paddedWidth = bottomWidth + 2 * padSize
      let boundKernelPositionY = (paddedHeight - kernelSize + stride) / stride * stride
      let boundKernelPositionX = (paddedWidth - kernelSize + stride) / stride * stride

      for i in 0 ..< batchSize {
        output[i].reset(0)
      }

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
                      if bottom[currentBatch][currentChannel, row - padSize, col - padSize] > pooled {
                        pooled = bottom[currentBatch][currentChannel, row - padSize, col - padSize]
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
                      pooled = bottom[currentBatch][currentChannel, row - padSize, col - padSize] / (Tensor.DataType(kernelSize * kernelSize))
                    }
                  }
                }
                output[currentBatch][currentChannel, kernelPositionY / stride, kernelPositionX / stride] += pooled
              }
            }
          }
        }
      }
    }
  }

  func layerSetUp(networkProperties: NetworkProperties, bottomNumOutput: Int? = nil) {
    self.engine = networkProperties.engine
    self.myNumOutput = bottomNumOutput!
    // Set batch size
    for _ in 0 ..< networkProperties.batchSize {
      self.output.append(Tensor())
      self.gradient.append(Tensor())
    }
  }

  func reshape(bottomDimensionsOpt: [Int]?) {
    if let bottomDimensionsOpt = bottomDimensionsOpt {
      for i in self.output.indices {
        self.output[i].reshape(bottomDimensionsOpt)
        self.gradient[i].reshape(bottomDimensionsOpt)
      }
    }
  }

  func numOutput() -> Int {
    // When?
    return myNumOutput
  }

  func forwardGPU(bottomOpt: [Tensor]?) {}

  func backwardCPU(topOpt: [Tensor]?) {}
  func backwardGPU(topOpt: [Tensor]?) {}
}

public struct PoolingParameters: LayerParameterProtocol {

  public enum PoolingMethod {
    case MAX
    case AVG
  }

  public let kernelSize : Int
  public let stride : Int
  public let padSize : Int
  public let method : PoolingMethod
  public init(kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
              method: PoolingMethod = .MAX) {
    self.kernelSize = kernelSize
    self.stride = stride
    self.padSize = padSize
    self.method = method
  }
}