//
//  ReluLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief ReLU layer.
 */
public class ReluLayer: ForwardBackwardLayer {
    let name:String="ReLU Layer"
    var data: Tensor<Int>
    init(data: Tensor<Int>) {
        self.data = data
    }
}