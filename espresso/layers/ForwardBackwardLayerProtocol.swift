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

extension ForwardLayerProtocol {
  func reshape(bottomDimensionsOpt: [Int]?) {
    // Reshape the output (and gradient)
  }
}