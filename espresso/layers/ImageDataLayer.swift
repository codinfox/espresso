//
//  ImageDataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The image data input layer.
 */
public class ImageDataLayer : DataLayerProtocol {
  public var name: String
  public var output: [Tensor]
  public var batchNo:Int
  public var engine: NetworkProperties.NetworkEngine
  var parameters: ImageDataParameters
  var channelNo: Int
  var height: Int
  var width: Int

  public init(name:String, parameters:ImageDataParameters) {
    self.name = name
    self.parameters = parameters
    self.batchNo = 0
    self.parameters = parameters
    self.channelNo = parameters.dimensions[0]
    self.height = parameters.dimensions[1]
    self.width = parameters.dimensions[2]
    self.output = []
    self.engine = .CPU
  }

  func forwardCPU(bottom: [Tensor]?) {
    let imgSize = self.height * self.width
    let batchSize = 1
    let start = batchNo * batchSize
    for i in 0..<batchSize {
      let data = parameters.readImage(parameters.imgNames[start + i])
      let trainData:[Float] = data.0
      // let trainLabel = data.1 //(TODO) Later
      output[i].storage.replaceRange(i*imgSize..<(i+1)*imgSize, with: trainData)
    }
  }
  
  func forwardGPU(bottom: [Tensor]?) {
    forwardCPU(bottom)
  }
  
  func reshape(bottomDimensions: [Int]?) {
    if bottomDimensions != nil {
      let dimensions = bottomDimensions!
      self.channelNo = dimensions[0]
      self.height = dimensions[1]
      self.width = dimensions[2]
      let batchSize = 1
      for i in 0..<batchSize {
        self.output[i].reshape(dimensions)
      }
    }
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
