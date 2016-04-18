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
public class ImageDataLayer : DataLayerProtocol {
  public var name: String
  public var output: Tensor = Tensor()
  public var batchNo:Int
  public var engine: NetworkProperties.NetworkEngine
  public var metalDevice: MTLDevice!
  public var metalCommandQueue: MTLCommandQueue!
  public var metalDefaultLibrary: MTLLibrary!
  var parameters: ImageDataParameters

  public init(name:String, parameters:ImageDataParameters) {
    self.name = name
    self.parameters = parameters
    self.batchNo = 0
    self.parameters = parameters
    self.engine = .CPU
  }

  func forwardCPU(bottom: Tensor?) {
    //    let imgSize = parameters.dimensions[1] * parameters.dimensions[2]
    //    let batchSize = 1
    //    let start = batchNo * batchSize
    //    if start > parameters.imgNames.count {
    //      print("error: not enough images")
    //    }
    //    for curBatch in 0..<batchSize {
    //      let data = parameters.readImage(parameters.imgNames[start + curBatch])
    //      let trainData:[Float] = data.0
    //      // let trainLabel = data.1 //(TODO) Later
    //      output[curBatch].storage.replaceRange(curBatch*imgSize..<(curBatch+1)*imgSize, with: trainData)
    //    }
    //    batchNo += 1
  }

  func forwardGPU(bottom: Tensor?) {
    let imgSize = parameters.dimensions[2] * parameters.dimensions[3]
    let batchSize = parameters.dimensions[0]
    let start = batchNo * batchSize
    if start > parameters.imgNames.count {
      print("error: not enough images")
    }
    for curBatch in 0..<batchSize {
      let data = parameters.readImage(parameters.imgNames[start + curBatch])
      let trainData:[Float] = data.0
      output.storage.replaceRange(curBatch*imgSize..<(curBatch+1)*imgSize, with: trainData)
      output.mtlStorage = createFloatArray(output.storage, metalDevice: metalDevice)
    }
    batchNo += 1
  }

  func reshape(bottomDimensionsOpt: [Int]?) {
    //    if bottomDimensionsOpt != nil {
    //      let dimensions = bottomDimensionsOpt!
    //      let batchSize = 1
    //      for curBatch in 0..<batchSize {
    //        if self.output.count <= curBatch {
    //          self.output.append(Tensor(dimensions: dimensions))
    //        } else {
    //          self.output[curBatch].reshape(dimensions)
    //        }
    //      }
    //    }
  }

  public func layerSetUp(networkProperties: NetworkProperties) {

  }
}

public struct ImageDataParameters: LayerParameterProtocol {
  public var imgNames: [String]
  public var dimensions:[Int]
  public var readImage: String->([Float], [Float])
  public init(imgNames: [String], dimensions: [Int], readImage: String->([Float], [Float])) {
    self.imgNames = imgNames
    self.dimensions = dimensions
    self.readImage = readImage
  }
}
