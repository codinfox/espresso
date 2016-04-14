//
//  ConvolutionLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Convolution layer.
 */
public class ConvolutionLayer: ForwardBackwardLayer {
    let name:String="Convolution Layer"
    var data: Tensor<Int>
    init(data: Tensor<Int>) {
        self.data = data
    }
}