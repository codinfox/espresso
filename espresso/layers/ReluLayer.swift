//
//  ReluLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief ReLU layer.
 */
public class ReluLayer: ForwardBackwardLayerProtocol {
  public var name : String
  var parameters : ReLUParameters
  public var output: [Tensor] = []
  public var gradient: [Tensor] = []
  public var engine: NetworkProperties.NetworkEngine = .CPU

  public init(name: String = "relu", parameters: ReLUParameters) {
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

      for currentBatch in 0 ..< batchSize {
        for i in bottom[currentBatch].storage.indices {
          output[currentBatch].storage[i] = max(0, bottom[currentBatch].storage[i]) + parameters.negativeSlope * min(0, bottom[currentBatch].storage[i])
        }
      }
    }
  }

  func forwardGPU(bottomOpt: [Tensor]?) {}

  func backwardCPU(topOpt: [Tensor]?) {}
  func backwardGPU(topOpt: [Tensor]?) {}


}

public struct ReLUParameters : LayerParameterProtocol {
  public var negativeSlope : Tensor.DataType
  public init(negativeSlope: Tensor.DataType = 0) {
    self.negativeSlope = negativeSlope
  }
}