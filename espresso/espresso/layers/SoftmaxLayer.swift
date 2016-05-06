//
//  Softmax.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal
import Accelerate

/** @brief Softmax layer.
 This can also be not backwardable
 */
public class SoftmaxLayer: ForwardLayerProtocol, BackwardLayerProtocol {
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
  var parameters : SoftmaxParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  private var myNumOutput: Int = 0

  public init(parameters: SoftmaxParameters) {
    self.parameters = parameters
  }

  func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
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
    //    self.gradient.reshape(oneBottomDimensionsSample)
  }

  func forwardCPU(bottom: [Tensor]) {
    if bottom.count > 0 {
      let bottom = bottom[0] // in softmax layer, bottom is really just a single Tensor

      // how many bins does an output distribution has
      let numOutput = bottom.dimensions[self.parameters.axis]
      // for conv feature maps, this is just height * width
      let mapSizeToPerformOn = bottom.count(fromDimension: self.parameters.axis + 1)
      // totally how many distributions should we get
      let totalNumberOfDistributions = bottom.count(toDimension: self.parameters.axis - 1)

      let batchElements = numOutput * mapSizeToPerformOn
      let allOnes = [Float](count: batchElements, repeatedValue: 1)

      for i in 0..<bottom.count() {
        self.output.storage[i] = bottom.storage[i]
      }
      /**
       *  For the typical 4-dimensional case [batchSize, channel, height, width], the typical axis to perform softmax on is 1, which is the channel dimension:
       *      `totalNumberOfDistributions` == batchSize,
       *      `numOutput` == channels,
       *      `mapSizeToPerformOn` == height * width
       */
      for mapIndex in 0 ..< totalNumberOfDistributions {
        // The max pixel across all the channels(parameters.axis)
        var maxPixels = [Float](count: mapSizeToPerformOn, repeatedValue: -Tensor.DataType.infinity)
        for gridIndex in 0..<mapSizeToPerformOn {
          for currentBin in 0..<numOutput {
            let index = mapIndex * numOutput * mapSizeToPerformOn + currentBin * mapSizeToPerformOn + gridIndex
            maxPixels[gridIndex] = max(maxPixels[gridIndex], bottom.storage[index])
          }
        }

        //var mapMatrix = [Float](count: batchElements, repeatedValue: 0)
        // subtraction
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(numOutput), Int32(mapSizeToPerformOn), 1, -1, allOnes, 1, &maxPixels, Int32(mapSizeToPerformOn), 1, &self.output.storage + mapIndex * batchElements, Int32(mapSizeToPerformOn))

        var expRes = [Float](count: batchElements, repeatedValue: 0)
        var elements:Int32 = Int32(batchElements)
        // exp
        vvexpf(&expRes, &self.output.storage, &elements)

        // sum
        cblas_sgemv(CblasRowMajor, CblasTrans, Int32(numOutput), Int32(mapSizeToPerformOn), 1, &expRes, Int32(mapSizeToPerformOn), allOnes, 1, 0, &maxPixels, 1)
//        var scalar:Float = 1

        for i in 0..<bottom.count() {
          self.output.storage[i] = expRes[i]
        }
        // debug point
        for currentBin in 0..<numOutput {
          vDSP_vdiv(&maxPixels, 1, &expRes + (currentBin * mapSizeToPerformOn), 1, &self.output.storage + (currentBin * mapSizeToPerformOn), 1, vDSP_Length(mapSizeToPerformOn))
        }
      }
    }
  }

  func forwardGPU(bottom: [Tensor]) {
    if (bottom.count > 0) {
      let bottom = bottom[0]
      let commandBuffer = self.metalCommandQueue.commandBuffer()
      // how many bins does an output distribution has
      let numOutput = bottom.dimensions[self.parameters.axis]
      // for conv feature maps, this is just height * width
      let mapSizeToPerformOn = bottom.count(fromDimension: self.parameters.axis + 1)
      // totally how many distributions should we get
      let totalNumberOfDistributions = bottom.count(toDimension: self.parameters.axis - 1)

      // copy the parameters to metal
      let paramBuffer = createSoftmaxParameter(MetalSoftmaxParameter(numOutput: numOutput, totalNumberOfDistributions: totalNumberOfDistributions, mapSizeToPerformOn: mapSizeToPerformOn), metalDevice: metalDevice)
      // perform computation
      submitCommonComputeJob("softmaxForward", paramBuffer: paramBuffer, metalDefaultLibrary: self.metalDefaultLibrary, metalDevice: self.metalDevice, inputData: bottom, outputData: self.output, commandBuffer: commandBuffer)
    }
  }
}

public struct SoftmaxParameters : LayerParameterProtocol {
  public let name : String
  public let dependencies: [String]
  /// Perform Softmax on which axis (usually 1, the channel axis). For example, FC layers, if have 1000 outputs (1000 channels), we perform on these 1000 channels and get 1000 probabilities (distribution)
  public let axis : Int
  public init(name: String,
              dependencies: [String],
              axis: Int = 1) {
    self.name = name
    self.dependencies = dependencies
    self.axis = axis
  }
}
