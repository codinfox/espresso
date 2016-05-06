//
//  Layer.swift
//  espresso
//
//  Created by Zhihao Li on 4/13/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief The base layer.
 */
protocol LayerProtocol {
  var name : String { get }
  var dependencies : [String] { get }
  var metalDevice: MTLDevice! { get set }
  var metalCommandQueue: MTLCommandQueue! { get set }
  var metalDefaultLibrary: MTLLibrary! { get set }

  //!!! API subject to change
  /**
   Setup current layer when added to the network

   This is a life-cycle method, will be called once added to the network. This method will take parameters from the network to do necessary initialization.
   This method is transparent to user and should be declared internal.
   */
  mutating func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                           bottomDimensions: [[Int]],
                           metalDevice: MTLDevice!,
                           metalDefaultLibrary: MTLLibrary!,
                           metalCommandQueue: MTLCommandQueue!)
}

protocol LayerParameterProtocol {
  var name: String { get }
  var dependencies: [String] { get }
}