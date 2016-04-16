//
//  SoftmaxWithLossLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Softmax With Loss layer.
 */
public class SoftmaxWithLossLayer: ForwardLayerProtocol, BackwardLayerProtocol {
  public var name : String
  public init(name: String = "softmaxwloss") {
    self.name = name
  }
}