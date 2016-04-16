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

  /**
   Setup current layer when added to the network
   
   This is a life-cycle method, will be called once added to the network. This method will take parameters from the network to do necessary initialization.
   This method is transparent to user and should be declared internal.
   */
  func layerSetUp(networkProperties: NetworkProperties)
}

extension LayerProtocol {
  /**
   Default layerSetUp method, should be called everytime when calling layerSetUp
   
   Class/Struct conforming to this protocol should not override this method, and when calling layerSetUp, one should always call superLayerSetUp inside layerSetUp first

   - parameter networkProperties:
   */
  mutating func superLayerSetUp(networkProperties: NetworkProperties) {
    self.engine = networkProperties.engine
  }

  /**
   Don't forget to call superLayerSetUp first

   - parameter networkProperties: 
   */
  mutating func layerSetUp(networkProperties: NetworkProperties) {
    superLayerSetUp(networkProperties)
    // TODO: do setup
  }
}

protocol LayerParameterProtocol {
  
}