//
//  LrnLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief LRN layer.
 */
public class LRNLayer: ForwardLayerProtocol, BackwardLayerProtocol {
  public var name : String
  var parameters : LRNParameters

  public init(name: String = "lrn", parameters: LRNParameters) {
    self.name = name
    self.parameters = parameters
  }
  // Implement protocols
}

public struct LRNParameters : LayerParameterProtocol {

  public enum NormRegion {
    case ACROSS_CHANNELS
    case WITHIN_CHANNEL
  }

  public let localSize: Int
  public let alpha: Float
  public let beta: Float
  public let region: NormRegion

  public init(localSize: Int = 5,
              alpha: Float = 1,
              beta: Float = 0.75,
              region: NormRegion = .ACROSS_CHANNELS) {
    self.localSize = localSize
    self.alpha = alpha
    self.beta = beta
    self.region = region
  }
}