//
//  ForwardBackwardLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

typealias BackwardLayerMethodType = ([Tensor]?)->()
/** @brief Protocol for Backward Layers
 */
protocol BackwardLayerProtocol : LayerProtocol {
  var gradient: Tensor { get set }
  var backwardMethod: BackwardLayerMethodType? { get set }

  /**
   This method does not need to be implemented in the code.
   */
  mutating func backward(top: [Tensor]?)
  mutating func backwardCPU(top: [Tensor]?)
  mutating func backwardGPU(top: [Tensor]?)
}

extension BackwardLayerProtocol {
  mutating func backward(top: [Tensor]?) {
    self.backwardMethod!(top)
  }
  mutating func backwardCPU(top: [Tensor]?) {
    // Do nothing
  }
  mutating func backwardGPU(top: [Tensor]?) {
    // Do nothing
  }
}