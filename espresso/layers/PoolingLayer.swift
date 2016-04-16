//
//  PollingLayer.swift
//  espresso
//
//  Created by Zhihao Li on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Polling layer.
 */
public class PoolingLayer: ForwardLayerProtocol, BackwardLayerProtocol {
  public var name : String

  var parameters : PoolingLayer

  public init(name: String = "pooling", parameters: PoolingLayer) {
    self.name = name
    self.parameters = parameters
  }
}

public struct PoolingParameters: LayerParameterProtocol {

  public enum PoolingMethod {
    case MAX
    case AVG
  }

  public let kernelSize : Int
  public let stride : Int
  public let padSize : Int
  public let method : PoolingMethod
  public init(kernelSize: Int,
              stride: Int = 1,
              padSize: Int = 0,
              method: PoolingMethod = .MAX) {
    self.kernelSize = kernelSize
    self.stride = stride
    self.padSize = padSize
    self.method = method
  }
}