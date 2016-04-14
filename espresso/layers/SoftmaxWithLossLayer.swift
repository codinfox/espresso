//
//  SoftmaxWithLossLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Softmax With Loss layer.
 */
public class SoftmaxWithLossLayer: ForwardBackwardLayer {
    let name:String="Softmax With Loss Layer"
    var data: Tensor<Int>
    init(data: Tensor<Int>) {
        self.data = data
    }
}