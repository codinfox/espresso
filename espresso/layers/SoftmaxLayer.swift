//
//  Softmax.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Softmax layer.
 This can also be not backwardable
 */
public class SoftmaxLayer: ForwardLayerProtocol, BackwardLayerProtocol {
  public var name : String
  public init(name: String = "softmax") {
    self.name = name
  }
}