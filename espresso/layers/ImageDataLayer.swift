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
public class ImageDataLayer : ForwardLayerProtocol {
  public var name: String
  public var output: [Tensor]
  public var batchNo:Int
  var parameters: ImageDataParameters
  var batchSize: Int
  var channelNo: Int
  var height: Int
  var width: Int
  let isCpu: Bool
  public init(name:String, parameters:ImageDataParameters) {
    self.name = name
    self.parameters = parameters
    self.batchNo = 0
    self.parameters = parameters
    self.batchSize = parameters.dimensions[0]
    self.channelNo = parameters.dimensions[1]
    self.height = parameters.dimensions[2]
    self.width = parameters.dimensions[3]
    self.output = []
    self.isCpu = parameters.isCpu
  }

  func forward_cpu(bottom: [Tensor]?) {
    let imgSize = self.height * self.width
    let start = batchNo * batchSize
    for i in 0..<batchSize {
      let data = parameters.readImage(parameters.imgNames[start + i])
      let trainData:[Int] = data.0
      // let trainLabel = data.1 //(TODO) Later
      output[i].storage.replaceRange(i*imgSize..<(i+1)*imgSize, with: trainData)
    }
  }
  
  func forward_gpu(bottom: [Tensor]?) {
    forward_cpu(bottom)
  }
  
  func reshape(bottomDimensions: [Int]?) {
    if bottomDimensions != nil {
      let dimensions = bottomDimensions!
      self.batchSize = dimensions[0]
      self.channelNo = dimensions[1]
      self.height = dimensions[2]
      self.width = dimensions[3]
      self.output = []
    }
  }
}

public struct ImageDataParameters: LayerParameterProtocol {
  public var imgNames: [String]
  public var dimensions:[Int]
  public var readImage: String->([Int], [Int])
  public var isCpu : Bool
  public init(imgNames: [String], dimensions: [Int], readImage: String->([Int], [Int]), isCpu: Bool) {
    self.imgNames = imgNames
    self.dimensions = dimensions
    self.readImage = readImage
    self.isCpu = isCpu
  }
}
