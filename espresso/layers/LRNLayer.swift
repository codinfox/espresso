//
//  LrnLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief LRN layer.
 */
public class LRNLayer: ForwardBackwardLayerProtocol {
  public var name : String
  public var output: [Tensor] = []
  public var gradient: [Tensor] = []
  public var engine: NetworkProperties.NetworkEngine = .CPU
  private var myNumOutput : Int = 0
  var parameters : LRNParameters

  public init(name: String = "lrn", parameters: LRNParameters) {
    self.name = name
    self.parameters = parameters
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
    return myNumOutput
  }

  func forwardCPU(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0){
      let bottom = bottomOpt!
      let batchSize = bottom.count

      let bottomChannels = bottom[0].dimensions[0]
      let bottomHeight = bottom[0].dimensions[1]
      let bottomWidth = bottom[0].dimensions[2]

      for currentBatch in 0 ..< batchSize {
        switch parameters.region {
        case .ACROSS_CHANNELS:
          for y in 0 ..< bottomHeight {
            for x in 0 ..< bottomWidth {
              for currentChannel in 0 ..< bottomChannels {
                var Z : Tensor.DataType = 0
                for maskIndex in 0 ..< parameters.localSize {
                  let currentMask = currentChannel - parameters.localSize / 2 + maskIndex
                  Z += bottom[currentBatch][currentMask, y, x] * bottom[currentBatch][currentMask, y, x]
                }
                output[currentBatch][currentChannel, y, x] = pow(1 + parameters.alpha / Tensor.DataType(parameters.localSize * parameters.localSize) / Z, parameters.beta)
              }
            }
          }
        case .WITHIN_CHANNEL:
          for currentChannel in 0 ..< bottomChannels {
            for y in 0 ..< bottomHeight {
              for x in 0 ..< bottomWidth {
                let maskY = y - parameters.localSize
                let maskX = x - parameters.localSize
                var Z : Tensor.DataType = 0
                for kernelY in 0 ..< parameters.localSize {
                  if maskY + kernelY < 0 || maskY + kernelY >= bottomHeight {
                    continue
                  }
                  for kernelX in 0 ..< parameters.localSize {
                    if maskX + kernelX < 0 || maskY + kernelY >= bottomWidth {
                      continue
                    }
                    Z += bottom[currentBatch][currentChannel, maskY + kernelY, maskX + kernelX] * bottom[currentBatch][currentChannel, maskY + kernelY, maskX + kernelX]
                  }
                }
                output[currentBatch][currentChannel, y, x] = pow(1 + parameters.alpha / Tensor.DataType(parameters.localSize * parameters.localSize) / Z, parameters.beta)
              }
            }
          }
        }
      }
    }
  }

  func forwardGPU(bottomOpt: [Tensor]?) {}

  func backwardCPU(topOpt: [Tensor]?) {}
  func backwardGPU(topOpt: [Tensor]?) {}

}

public struct LRNParameters : LayerParameterProtocol {

  public enum NormRegion {
    case ACROSS_CHANNELS
    case WITHIN_CHANNEL
  }

  public let localSize: Int
  public let alpha: Float
  public let beta: Float
  public let region: NormRegion

  public init(localSize: Int = 5,
              alpha: Float = 1,
              beta: Float = 0.75,
              region: NormRegion = .ACROSS_CHANNELS) {
    self.localSize = localSize
    self.alpha = alpha
    self.beta = beta
    self.region = region
  }
}