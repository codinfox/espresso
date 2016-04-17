//
//  ForwardBackwardLayerProtocol.swift
//  espresso
//
//  Created by Zhihao Li on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

protocol ForwardBackwardLayerProtocol : ForwardLayerProtocol, BackwardLayerProtocol {

}

extension ForwardBackwardLayerProtocol {
  mutating func reshape(bottomDimensionsOpt: [Int]?) {
    if bottomDimensionsOpt != nil {
      let dimensions = bottomDimensionsOpt!
      let batchSize = 1
      for i in 0..<batchSize {
        if self.output.count <= i {
          assert(self.output.count == self.gradient.count, "output and gradient numbers not match")
          self.output.append(Tensor(dimensions: dimensions))
          /* gradient?? */
          self.gradient.append(Tensor(dimensions: dimensions))
        } else {
          self.output[i].reshape(dimensions)
          self.gradient[i].reshape(dimensions)
        }
      }
    }
  }
}