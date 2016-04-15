//
//  ConvolutionLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Convolution layer.
 */
public class ConvolutionLayer<DataType>: ForwardBackwardLayer {
  public let name:String="Convolution Layer"
  public var output : Tensor<DataType>
  public var gradient : Tensor<DataType>

  var data: Tensor<DataType>
  init(data: Tensor<DataType>) {
    self.data = data
  }

  public func forward_cpu(input: Tensor<DataType>) {
  
  }
  public func backward_cpu(input: Tensor<DataType>) {
  
  }
}

public struct ForwardBackwardLayerParams: Parameter {
  
}