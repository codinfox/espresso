//
//  DataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The input data layer.
 */
public protocol DataLayer : Layer {
  associatedtype DataType
  var output:Tensor<DataType> {get set}
  func forward_cpu()
  func forward_gpu()
}