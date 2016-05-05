//
//  ReluLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal
import Accelerate
import simd

/** @brief ReLU layer.
 */
public class ReluLayer: ForwardLayerProtocol, BackwardLayerProtocol {
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
  var parameters : ReLUParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  public init(parameters: ReLUParameters) {
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
    self.reshapeByBottomDimensions(bottomDimensions) // may exception (should not)
  }

  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
    let oneBottomDimensionsSample = bottomDimensions[0]

    self.output.reshape(oneBottomDimensionsSample)
    self.gradient.reshape(oneBottomDimensionsSample)
  }

  func forwardCPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0] // in softmax layer, bottom is really just a single Tensor
      for index in 0 ..< bottom.numel {
        output.storage[index] = max(0, bottom.storage[index]) + self.parameters.negativeSlope * min(0, bottom.storage[index])
      }
//      output.storage = bottom.storage.map({ max(0, $0) + self.parameters.negativeSlope * min(0, $0)})
    }
  }

  func forwardGPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0]
      let commandBuffer = self.metalCommandQueue.commandBuffer()
      // copy the parameters to metal
      let paramBuffer = createReluParam(MetalReluParameter(negativeSlope: self.parameters.negativeSlope), metalDevice: metalDevice)
      // perform computation
      submitCommonComputeJob("reluForward", paramBuffer: paramBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice, inputData: bottom, outputData: self.output, commandBuffer: commandBuffer)
    }
  }
}

public struct ReLUParameters : LayerParameterProtocol {
  public let name : String
  public let dependencies: [String]
  public var negativeSlope : Tensor.DataType
  public init(name: String,
              dependencies: [String],
              negativeSlope: Tensor.DataType = 0) {
    self.negativeSlope = negativeSlope
    self.name = name
    self.dependencies = dependencies
  }
}
