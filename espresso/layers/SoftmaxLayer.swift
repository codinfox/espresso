//
//  Softmax.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Softmax layer.
 This can also be not backwardable
 */
public class SoftmaxLayer: ForwardBackwardLayerProtocol {
  public var name : String
  public var output: [Tensor] = []
  public var gradient: [Tensor] = []
  public var engine: NetworkProperties.NetworkEngine = .CPU
  var parameters : SoftmaxParameters

  public init(name: String = "softmax", parameters: SoftmaxParameters) {
    self.name = name
    self.parameters = parameters
  }

  func layerSetUp(networkProperties: NetworkProperties, bottomNumOutput: Int? = nil) {
    self.engine = networkProperties.engine
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
    return parameters.numOutput
  }

  func forwardCPU(bottomOpt: [Tensor]?) {
    if bottomOpt != nil && (bottomOpt!.count > 0){
      let bottom = bottomOpt!
      let batchSize = bottom.count

      let bottomChannels = bottom[0].dimensions[0]
      let bottomHeight = bottom[0].dimensions[1]
      let bottomWidth = bottom[0].dimensions[2]

      for i in 0..<batchSize {
        output[i].reset(0)
      }

      for currentBatch in 0 ..< batchSize {
        for y in 0 ..< bottomHeight {
          for x in 0 ..< bottomWidth {
            var Z : Tensor.DataType = 0
            for currentChannel in 0 ..< bottomChannels {
              let currentTerm = exp(bottom[currentBatch][currentChannel, y, x])
              output[currentBatch][currentChannel, y, x] = currentTerm
              Z += currentTerm
            }
            for currentChannel in 0 ..< bottomChannels {
              output[currentBatch][currentChannel, y, x] /= Z
            }
          }
        }
      }
    }
  }

  func forwardGPU(bottomOpt: [Tensor]?) {
  }

  func backwardCPU(topOpt: [Tensor]?) {}
  func backwardGPU(topOpt: [Tensor]?) {}

}

public struct SoftmaxParameters : LayerParameterProtocol {
}
