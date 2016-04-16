//
//  DropoutLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Dropout layer.
 */
public class DropoutLayer: ForwardBackwardLayerProtocol {
  public var name : String
  var parameters : DropoutParameters

  public init(name: String = "dropout", parameters: DropoutParameters) {
    self.name = name
    self.parameters = parameters
  }
  // Implement protocols
}

public struct DropoutParameters : LayerParameterProtocol {
  public let dropoutRatio : Float
  public init(dropoutRatio: Float = 0.5) {
    self.dropoutRatio = dropoutRatio
  }
}