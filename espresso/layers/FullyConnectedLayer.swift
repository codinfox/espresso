//
//  FullyConnectedLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Fully connected layer.
 */
public class FullyConnectedLayer: ForwardBackwardLayer {
    let name:String="Fully Connected Layer"
    var data: Tensor<Int>
    init(data: Tensor<Int>) {
        self.data = data
    }
}