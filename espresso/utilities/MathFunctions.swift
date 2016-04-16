//
//  MathFunctions.swift
//  espresso
//
//  Created by Jerry Zhang on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

public func sigmoid_cpu(x: Float) -> Float {
    return 1.0 / (1.0 + exp(x))
}
