//
//  DropoutLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Dropout layer.
 */
public class DropoutLayer: ForwardBackwardLayer {
    let name:String="Dropout Layer"
    var data: Tensor<Int>
    init(data: Tensor<Int>) {
        self.data = data
    }
}