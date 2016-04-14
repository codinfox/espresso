//
//  DataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The input data layer.
 */
protocol DataLayer : Layer {
    var data:Tensor<Int> {get}
    var parameters:[String:Double] {get set}
}