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
    // invalid default reshape implementation
    if let dimensions = bottomDimensionsOpt {
      for i in self.output.indices {
        self.output[i].reshape(dimensions)
        self.gradient[i].reshape(dimensions)
      }
    }
  }
}