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
  public var output: Tensor
  public var batchNo:Int
  var parameters:ImageDataParameters
  var batchSize:Int
  var channelNo:Int
  var height:Int
  var width:Int
  public init(name:String, parameters:ImageDataParameters) {
    self.name = name
    self.parameters = parameters
    self.batchNo = 0
    self.parameters = parameters
    self.batchSize = parameters.dimensions[0]
    self.channelNo = parameters.dimensions[1]
    self.height = parameters.dimensions[2]
    self.width = parameters.dimensions[3]
  }
  
  
  public func forward(bottom: Tensor?) {
    
  }
  
  public func forward_cpu(bottom: Tensor?) {
    output = Tensor(dimensions: parameters.dimensions)
    for i in (batchNo * batchSize)..<((batchNo + 1) * batchSize) {
      output.storage.appendContentsOf(parameters.readImage(parameters.imgNames[i]))
    }
  }
  
  public func forward_gpu(bottom: Tensor?) {
    forward_cpu(bottom)
  }
  
  public func reshape(dimensions: [Int]) {
    
  }
}

public struct ImageDataParameters: LayerParameterProtocol {
  public var imgNames: [String]
  public var dimensions:[Int]
  public var resize : [Int]?
  public var isGrey : Bool
  public var readImage: String->([Int], [Int])
  public init(imgNames: [String], dimensions: [Int], resize: [Int]?,
              isGrey: Bool, readImage: String->([Int], [Int])) {
    self.imgNames = imgNames
    self.dimensions = dimensions
    self.resize = resize
    self.isGrey = isGrey
  }
}
