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
  public var output: Tensor = Tensor()
  public var gradient: Tensor = Tensor()
  public var weight: Tensor = Tensor()
  public var bias: Tensor = Tensor()
  var parameters : ReLUParameters
  var negativeSlope : Tensor.DataType
  public var engine: NetworkProperties.NetworkEngine
  public var metalDevice: MTLDevice!
  public var metalCommandQueue: MTLCommandQueue!
  public var metalDefaultLibrary: MTLLibrary!
  public var commandBufferQueue: [MTLCommandBuffer]!

  func forwardCPU(bottomOpt: Tensor?) {
//    if bottomOpt != nil && (bottomOpt!.count > 0){
//      let bottom = bottomOpt!
//      let batchSize = bottom.count
//      let channels = bottom[0].dimensions[0]
//      let height = bottom[0].dimensions[1]
//      let width = bottom[0].dimensions[2]
//      for i in 0..<batchSize {
//        output[i].reset(0)
//      }
//      for i in 0..<batchSize {
//        for j in 0..<channels {
//          for k in 0..<height {
//            for l in 0..<width {
//              output[i][j,k,l] = max(0, bottom[i][j,k,l]) + negativeSlope * min(0, bottom[i][j,k,l])
//            }
//          }
//        }
//      }
//    }
  }

  func forwardGPU(bottomOpt: Tensor?) {
    let bottom = bottomOpt!
    // bottom tensor has been copied to GPU in previous layer
    let commandBuffer = self.metalCommandQueue.commandBuffer()
    // copy the parameters to metal
    let paramBuffer = createReluParam(MetalReluParameter(negativeSlope: self.negativeSlope, inputDim: bottom.dimensions), metalDevice: metalDevice)
    // perform computation
    /* should return commandBuffer? */
    submitComputeJob("reluForward", paramBuffer: paramBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice, inputData: bottom, outputData: self.output, commandBuffer: commandBuffer)
    commandBufferQueue.append(commandBuffer)
  }

  func backwardCPU(topOpt: Tensor?) {}
  func backwardGPU(topOpt: Tensor?) {}

  func initWeights() {
  }

  func updateWeights(weightGrad: Tensor){
  }

  func initBias() {}

  func updateBias(biasGrad: Tensor) {
  }

  public init(name: String = "relu", parameters: ReLUParameters) {
    self.name = name
    self.parameters = parameters
    self.negativeSlope = parameters.negativeSlope
    self.engine = .CPU
  }

  public func layerSetUp(networkProperties: NetworkProperties) {

  }

}

public struct ReLUParameters : LayerParameterProtocol {
  public var negativeSlope : Tensor.DataType
  public init(negativeSlope: Tensor.DataType = 0) {
    self.negativeSlope = negativeSlope
  }
}