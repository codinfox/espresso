//
//  LrnLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Accelerate
import Metal

/** @brief LRN layer.
 */
public class LRNLayer: ForwardLayerProtocol, BackwardLayerProtocol {
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
  private var myNumOutput : Int = 0
  var parameters : LRNParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  var compressedInfo: CompressedInfo! = nil

  public init(parameters: LRNParameters) {
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
    // self.gradient.reshape(oneBottomDimensionsSample)
  }

  func forwardCPU(bottom: [Tensor]) {
    if (bottom.count > 0) {
      let bottom = bottom[0]

      let bottomBatches = bottom.dimensions[0]
      let bottomChannels = bottom.dimensions[1]
      let bottomHeight = bottom.dimensions[2]
      let bottomWidth = bottom.dimensions[3]

      let numELementsPerBatch = bottomChannels * bottomHeight * bottomWidth
      let numElementsPerChan = bottomHeight * bottomWidth
      switch parameters.region {
      case .ACROSS_CHANNELS:
        // reusing paddedSquare in all batches
        var paddedSquare = Tensor(dimensions: [1, bottomChannels + parameters.localSize - 1, bottomHeight, bottomWidth])
        var scale = Tensor(dimensions: [bottomBatches, bottomChannels, bottomHeight, bottomWidth])
        let leftPadding = (parameters.localSize - 1) / 2 // local size is an odd number
        for currentBatch in 0..<bottomBatches {
          let batchOffset = currentBatch * numELementsPerBatch
          vDSP_vsq(&bottom.storage + batchOffset, 1, &paddedSquare.storage + (leftPadding * numElementsPerChan), 1, vDSP_Length(numELementsPerBatch))
          for local in 0..<parameters.localSize {
            cblas_saxpy(Int32(numElementsPerChan), parameters.alpha / Float(parameters.localSize), &paddedSquare.storage + (local * numElementsPerChan), 1, &scale.storage + currentBatch * numELementsPerBatch, 1)
          }
          for curChan in 1..<bottomChannels {
            // copy previous scale
            scale.storage.replaceRange((batchOffset + curChan * numElementsPerChan)..<(batchOffset + (curChan + 1) * numElementsPerChan), with: scale.storage[(batchOffset + (curChan-1) * numElementsPerChan)..<(batchOffset + curChan * numElementsPerChan)])
            // add head
            cblas_saxpy(Int32(numElementsPerChan), parameters.alpha / Float(parameters.localSize), &paddedSquare.storage + (curChan + parameters.localSize - 1) * numElementsPerChan, 1, &scale.storage + (currentBatch * numELementsPerBatch + curChan * numElementsPerChan), 1)
            // subtract tail
            cblas_saxpy(Int32(numElementsPerChan), -parameters.alpha / Float(parameters.localSize), &paddedSquare.storage + (curChan - 1) * numElementsPerChan, 1, &scale.storage + (currentBatch * numELementsPerBatch + curChan * numElementsPerChan), 1)
          }
          var minus_beta : Float = -parameters.beta
          var n = Int32(self.output.count())
          vvpowf(&self.output.storage, &minus_beta, &scale.storage, &n)
          vDSP_vmul(&bottom.storage, 1, &self.output.storage, 1, &self.output.storage, 1, vDSP_Length(self.output.count()))
        }
      case .WITHIN_CHANNEL:
        for currentBatch in 0..<bottomBatches {
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
                    Z += bottom[currentBatch, currentChannel, maskY + kernelY, maskX + kernelX] * bottom[currentBatch, currentChannel, maskY + kernelY, maskX + kernelX]
                  }
                }
                output[currentBatch, currentChannel, y, x] = bottom[currentBatch, currentChannel, y, x] / (pow(1 + parameters.alpha / Tensor.DataType(parameters.localSize * parameters.localSize) * Z, parameters.beta))
              }
            }
          }
        }
      }
    }
  }
  public func forwardGPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0]

      let bottomChannels = Int32(bottom.dimensions[1])
      let bottomHeight = Int32(bottom.dimensions[2])
      let bottomWidth = Int32(bottom.dimensions[3])

      let count = UInt32(self.output.count())
      let commandBuffer = self.metalCommandQueue.commandBuffer()
      // copy the parameters to metal
      let paramBuffer = createLrnParameter(MetalLrnParameter(count: count, localSize: Int32(parameters.localSize), bottomChannels: bottomChannels, bottomHeight: bottomHeight, bottomWidth: bottomWidth, alpha: parameters.alpha, beta: parameters.beta), metalDevice: self.metalDevice)
      var funcName = ""
      if (parameters.region == .ACROSS_CHANNELS) {
        funcName = "lrnCrossForward"
      } else {
        funcName = "lrnWithinForward"
      }

      // perform computation
      submitCommonComputeJob(funcName, paramBuffer: paramBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice, inputData: bottom, outputData: self.output, commandBuffer: commandBuffer, threadCount: self.output.count())
    }
  }
}

public struct LRNParameters : LayerParameterProtocol {

  public enum NormRegion {
    case ACROSS_CHANNELS
    case WITHIN_CHANNEL
  }

  public let name: String
  public let dependencies: [String]
  public let localSize: Int
  public let alpha: Float
  public let beta: Float
  public let region: NormRegion
  public init(name: String,
              dependencies: [String],
              localSize: Int = 5,
              alpha: Float = 0.0001,
              beta: Float = 0.75,
              region: NormRegion = .ACROSS_CHANNELS) {
    self.name = name
    self.dependencies = dependencies
    self.localSize = localSize
    self.alpha = alpha
    self.beta = beta
    self.region = region
  }
}