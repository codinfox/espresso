//
//  Net.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal


/** @brief Neural network.
 */
public class Network {
  var layers : [LayerProtocol]
  var parameters : NetworkProperties
  var layerDependencyMapping : [Int : [Int]]
  var layerNameIndexMapping : [String : Int]
  // Metal
  var metalDevice: MTLDevice!
  var metalDefaultLibrary: MTLLibrary!
  var metalCommandQueue: MTLCommandQueue!


  public init(parameters: NetworkProperties) {
    self.layers = []
    self.parameters = parameters
    self.layerDependencyMapping = [:]
    self.layerNameIndexMapping = [:]
    if parameters.engine == .GPU {
      // Initialize gpu
      metalDevice = MTLCreateSystemDefaultDevice()

      // Queue to handle an ordered list of command buffers
      metalCommandQueue = metalDevice.newCommandQueue()

      // Access to Metal functions that are stored in .metal file
      metalDefaultLibrary = metalDevice.newDefaultLibrary()
    }
  }

  public func add(layer: AnyObject) {
    var layer = layer as! LayerProtocol // make layer mutable
    var bottomDimensions : [[Int]] = []
    let currentLayerIndex = layers.count
    let currentLayerName = layer.name
    let currentLayerDependencies = layer.dependencies

    layerNameIndexMapping[currentLayerName] = currentLayerIndex
    layerDependencyMapping[currentLayerIndex] = []
    for depName in currentLayerDependencies {
      let depIndex = layerNameIndexMapping[depName]!
      layerDependencyMapping[currentLayerIndex]?.append(depIndex)
      bottomDimensions.appendContentsOf(((layers[depIndex] as! ForwardLayerProtocol).outputDimensions()))
    }

    self.layers.append(layer)
    layer.layerSetUp(engine: self.parameters.engine, bottomDimensions: bottomDimensions, metalDevice: self.metalDevice, metalDefaultLibrary: self.metalDefaultLibrary, metalCommandQueue: self.metalCommandQueue)
  }

  public func forward() {
    for index in self.layers.indices {
      var layer = self.layers[index] as! ForwardLayerProtocol // may exception, but should not

      var bottom : [Tensor] = []
      for dep in layerDependencyMapping[index]! {
        bottom.append((layers[dep] as! ForwardLayerProtocol).output)
      }

      layer.forward(bottom)
    }
  }

  public func backward() {
    // TODO
  }

  public func update() {
    // Update all the learnable parameters
    /**
     *  In Caffe, Network stores pointers to all the learnable parameters, outputs and gradients.
     */
  }

}

public struct NetworkProperties {

  public enum NetworkEngine {
    case GPU
    case CPU
  }

  public let batchSize : Int
  public let engine : NetworkEngine
  public init(batchSize: Int = 1,
              engine: NetworkEngine = .CPU) {
    self.batchSize = batchSize
    self.engine = engine
  }
}