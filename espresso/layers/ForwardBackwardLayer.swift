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
public protocol ForwardBackwardLayer: Layer {
    associatedtype DataType
    var output : Tensor<DataType> {get set}
    var gradient : Tensor<DataType> {get set}
    func forward_cpu(input: Tensor<DataType>)
    func backward_cpu(input: Tensor<DataType>)
}
