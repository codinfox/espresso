//
//  PollingLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Polling layer.
 */
public class PollyLayer: ForwardBackwardLayer {
    let name:String="Polling Layer"
    var data: Tensor<Int>
    init(data: Tensor<Int>) {
        self.data = data
    }
}