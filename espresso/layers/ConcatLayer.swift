//
//  ConcatLayer.swift
//  espresso
//
//  Created by Zhihao Li on 4/28/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

public class ConcatLayer: ForwardLayerProtocol, BackwardLayerProtocol {
  public var name : String {
    return parameters.name
  }

  public var dependencies: [String] {
    return self.parameters.dependencies
  }

  public var output: Tensor = Tensor()
  public var gradient: Tensor = Tensor()
  var parameters : ConcatParameters
  var forwardMethod: ForwardLayerMethodType? = nil
  var backwardMethod: BackwardLayerMethodType? = nil

  public init(parameters: ConcatParameters) {
    self.parameters = parameters
  }

  func layerSetUp(engine engine: NetworkProperties.NetworkEngine,
                         bottomDimensions: [[Int]]? = nil) {
    switch engine {
    case .CPU:
      self.forwardMethod = forwardCPU
    case .GPU:
      self.forwardMethod = forwardGPU
    }

    self.reshapeByBottomDimensions(bottomDimensions!) // may exception (should not)
  }

  func reshapeByBottomDimensions(bottomDimensions: [[Int]]) {
    var dimensions = bottomDimensions[0]

    for index in 1 ..< bottomDimensions.count {
      // test if dimensions are matched
      var tmpCurrent = bottomDimensions[index]
      var tmpOverall = dimensions
      tmpCurrent[self.parameters.axis] = 0
      tmpOverall[self.parameters.axis] = 0
      assert(tmpCurrent == tmpOverall, "concat input dimensions not match")

      dimensions[self.parameters.axis] += bottomDimensions[index][self.parameters.axis]
    }

    self.output.reshape(dimensions)
    //    self.gradient.reshape(dimensions) // may need to be handled differently
  }

  func forwardCPU(bottom: [Tensor]?) {
    if let bottom = bottom where bottom.count > 0 {
      let elementsBeforeTargetAxis = output.count(toDimension: self.parameters.axis - 1)

      var currentOutputCursor = 0
      for beforeIndex in 0 ..< elementsBeforeTargetAxis {
        for bottomIndex in bottom.indices {
          let bottomElementsInAndAfterTargetAxis = bottom[bottomIndex].count(fromDimension: self.parameters.axis)
          output.storage.replaceRange(
            currentOutputCursor ..< currentOutputCursor + bottomElementsInAndAfterTargetAxis,
            with: bottom[bottomIndex].storage[beforeIndex * bottomElementsInAndAfterTargetAxis ..< (beforeIndex + 1) * bottomElementsInAndAfterTargetAxis])
          currentOutputCursor += bottomElementsInAndAfterTargetAxis
        }
      }
      assert(currentOutputCursor == output.numel)
    }
  }

  func forwardGPU(bottom: [Tensor]?) {
    forwardCPU(bottom)
  }
}

public struct ConcatParameters: LayerParameterProtocol {
  public let name : String
  public let dependencies: [String]
  public let axis: Int // default 1, in the typical case, 1 represents channel
  public init(name: String,
              dependencies: [String],
              axis: Int = 1) {
    self.name = name
    self.dependencies = dependencies
    self.axis = axis
  }
}