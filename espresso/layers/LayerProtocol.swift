//
//  Layer.swift
//  espresso
//
//  Created by Zhihao Li on 4/13/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The base layer.
 */
public protocol LayerProtocol {
  var name: String { get set }
  var engine: NetworkProperties.NetworkEngine { get set }
//  var parameters : LayerParameterProtocol { get set }
//  init(name: String, parameters : LayerParameterProtocol)

}

extension LayerProtocol {
  /**
   Setup current layer when added to the network

   This is a life-cycle method, will be called once added to the network. This method will take parameters from the network to do necessary initialization.
   This method is transparent to user and should be declared internal.
   */

  func layerSetUp(networkProperties: NetworkProperties) {
    // TODO: do setup
  }
}

protocol LayerParameterProtocol {
  
}