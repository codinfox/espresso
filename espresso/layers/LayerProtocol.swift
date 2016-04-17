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

  //!!! API subject to change
  /**
   Setup current layer when added to the network

   This is a life-cycle method, will be called once added to the network. This method will take parameters from the network to do necessary initialization.
   This method is transparent to user and should be declared internal.

   - parameter networkProperties:
   - parameter bottomNumNeurons:  number of outputs in the bottom layer
   */
  mutating func layerSetUp(networkProperties: NetworkProperties, bottomNumOutput: Int? = nil) {
    self.engine = networkProperties.engine
    // TODO: do setup
  }

  /**
   The number of neurons in current layer

   For those layers that does not have actual "neurons" (e.g. ReLU layer), this is just the number of outputs (one feature map is counted as one output). For example, RGB image, the return value is 3
   - returns:
   */
  func numOutput() -> Int {
    return 1;
  }
}

protocol LayerParameterProtocol {
  
}