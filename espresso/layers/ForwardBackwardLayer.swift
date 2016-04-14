//
//  ForwardBackwardLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The layer that provides functions for forward and backward updates.
 */
protocol ForwardBackwardLayer: Layer {
    var data : Tensor<Int> {get set}
}