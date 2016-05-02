//
//  ImageDataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief The image data input layer.
 */
public class ImageDataLayer : ForwardLayerProtocol {
  public var name : String {
    return parameters.name
  }

  public var dependencies: [String] {
    return self.parameters.dependencies
  }

  public var output: Tensor = Tensor()
  public var batchNo:Int
  public var metalDevice: MTLDevice!
  public var metalCommandQueue: MTLCommandQueue!
  public var metalDefaultLibrary: MTLLibrary!

  var parameters: ImageDataParameters

  public init(parameters:ImageDataParameters) {
    self.parameters = parameters
    self.batchNo = 0
  }

  var forwardMethod: ForwardLayerMethodType? = nil

  func forwardCPU(bottom: [Tensor]) {
    let imgSize = parameters.dimensions[1] * parameters.dimensions[2] * parameters.dimensions[3]
    let batchSize = parameters.dimensions[0]
    let start = batchNo * batchSize
    if start > parameters.imgNames.count {
      print("error: not enough images")
    }
    for curBatch in 0..<batchSize {
      let data = parameters.readImage(parameters.imgNames[start + curBatch])
      let trainData:[Float] = data.0
      // let trainLabel = data.1 //(TODO) Later
      output.storage.replaceRange(curBatch*imgSize..<(curBatch+1)*imgSize, with: trainData)
    }
    batchNo += 1
  }
  
  func forwardGPU(bottom: [Tensor]) {
    let imgSize = parameters.dimensions[1] * parameters.dimensions[2] * parameters.dimensions[3]
    let batchSize = parameters.dimensions[0]
    let start = batchNo * batchSize
    if start > parameters.imgNames.count {
      print("error: not enough images")
    }
    for curBatch in 0..<batchSize {
      let data = parameters.readImage(parameters.imgNames[start + curBatch])
      let trainData:[Float] = data.0
      // let trainLabel = data.1 //(TODO) Later
      output.storage.replaceRange(curBatch*imgSize..<(curBatch+1)*imgSize, with: trainData)
    }
    // put the image array in metal buffer
    output.mtlStorage = createFloatArray(output.storage, metalDevice: metalDevice)
    batchNo += 1
  }

  func outputDimensions() -> [[Int]] {
    return [output.dimensions]
  }
  
  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
    let dimensions = parameters.dimensions
    self.output.reshape(dimensions)
   }

  func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                                bottomDimensions: [[Int]],
                                metalDevice: MTLDevice!,
                                metalDefaultLibrary: MTLLibrary!,
                                metalCommandQueue: MTLCommandQueue!) {
    switch engine {
    case .CPU:
      self.forwardMethod = forwardCPU
    case .GPU:
      self.forwardMethod = forwardGPU
    }
    self.metalDevice = metalDevice
    self.metalDefaultLibrary = metalDefaultLibrary
    self.metalCommandQueue = metalCommandQueue
    self.reshapeByBottomDimensions(bottomDimensions) // may exception (should not)
  }
}

public struct ImageDataParameters: LayerParameterProtocol {
  public var name: String
  public var imgNames: [String]
  public var dimensions:[Int] // batchSize, channel, height, width
  public var dependencies: [String]
  public var readImage: String->([Float], [Float])
  public init(name: String, imgNames: [String], dimensions: [Int], dependencies: [String], readImage: String->([Float], [Float])) {
    self.name = name
    self.imgNames = imgNames
    self.dimensions = dimensions
    self.readImage = readImage
    self.dependencies = dependencies
  }
}
