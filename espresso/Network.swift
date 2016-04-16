//
//  Net.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Neural network.
 */
public class Network {
  public var layers: [LayerProtocol]
  var parameters : NetworkProperties

  public init(parameters: NetworkProperties) {
    self.layers = []
    self.parameters = parameters
  }

  public func add(layer: LayerProtocol) {
    self.layers.append(layer)
    layer.layerSetUp(self.parameters)
  }

  public func forward() {
    // The first layer does not take in any input
    var bottomOutput : [Tensor]? = nil
    for l in self.layers {
      guard l is ForwardLayerProtocol else {
        break
      }
      let layer = l as! ForwardLayerProtocol
      layer.reshape(bottomOutput?[0].dimensions)
      layer.forward(bottomOutput)
      bottomOutput = layer.output
    }
  }

  public func backward() {
    // Last layer does not take any input
    var topGradient : [Tensor]? = nil
    for l in self.layers {
      guard l is BackwardLayerProtocol else {
        break
      }
      let layer = l as! BackwardLayerProtocol
      layer.backward(topGradient)
      topGradient = layer.gradient
    }
  }

  public func update() {
    // Update all the learnable parameters
    /**
     *  In Caffe, Network stores pointers to all the learnable parameters, outputs and gradients.
     */
  }

}

public struct NetworkProperties {
  public let batchSize : Int
  public init(batchSize: Int = 1) {
    self.batchSize = batchSize
  }
}