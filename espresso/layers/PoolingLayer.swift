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
  public var isCpu: Bool
  public var output: [Tensor]
  public var gradient: [Tensor]
  public var weight: Tensor
  public var bias: Tensor
  public var engine: NetworkProperties.NetworkEngine

  var parameters : PoolingParameters

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
        for j in 0..<channels {
          for k in 0.stride(to: padedHeight, by: stride) {
            for l in 0.stride(to: padedWidth, by: stride) {
              var pooled:Float = 0
              var count = 0
              if (parameters.method == .AVG) {
                for x in 0..<parameters.kernelSize {
                  for y in 0..<parameters.kernelSize {
                    let row = k + x
                    let col = l + y
                    if (row >= padSize && row < padedHeight - padSize
                      && col >= padSize && col < padedWidth) {
                      pooled += bottom[i][j, row, col]
                      count += 1
                    }
                  }
                }
                if count > 0 {
                  pooled = pooled / Float(count)
                }
              } else if (parameters.method == .MAX) {
                pooled = -Float.infinity
                for x in 0..<parameters.kernelSize {
                  for y in 0..<parameters.kernelSize {
                    let row = k + x
                    let col = l + y
                    if (row >= padSize && row < padedHeight - padSize
                      && col >= padSize && col < padedWidth) {
                      pooled = max(pooled, bottom[i][j, row, col])
                    }
                  }
                }
              }
              output[i][j, k, l] = pooled
            }
          }
        }
      }
    }
  }

  public func layerSetUp(networkProperties: NetworkProperties) {

  }

  func forward_gpu(bottomOpt: [Tensor]?) {}

  func backward_cpu(topOpt: [Tensor]?) {}
  func backward_gpu(topOpt: [Tensor]?) {}

  func reshape(bottomDimensionsOpt: [Int]?) {
    // Reshape the output (and gradient)
  }

  func initWeights() {
  }

  func updateWeights(weightGrad: Tensor){
  }

  func initBias() {}

  func updateBias(biasGrad: Tensor) {
  }

  public init(name: String = "pooling", parameters: PoolingParameters) {
    self.name = name
    self.parameters = parameters
  }


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