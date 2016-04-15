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
public class ImageDataLayer<DataType> : DataLayer {
  public var name: String
  public var output: Tensor<DataType>
  public var batchNo:Int
  var parameters:ImageDataLayerParams
  var batchSize:Int
  var channelNo:Int
  var height:Int
  var width:Int
  
  public init(name:String, parameters:ImageDataLayerParams) {
    self.name = name
    self.parameters = parameters
    self.output = Tensor<DataType>(dimensions: parameters.dimensions)
    self.batchNo = 0
  }
  
  public func forward_cpu() {
    for _ in 0..<batchSize {
      
    }
  }
  
  public func forward_gpu() {
    forward_cpu()
  }
}

public struct ImageDataLayerParams: Parameter {
  var imgFileName:String
  var dimensions:[Int]
  var resize : [Int]?
  var isGrey : Bool
  init(imgFileName: String, dimensions:[Int], resize:[Int]?, isGrey : Bool) {
    self.imgFileName = imgFileName
    self.dimensions = dimensions
    self.resize = resize
    self.isGrey = isGrey
  }
}
