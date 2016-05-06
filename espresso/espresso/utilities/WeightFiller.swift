//
//  WeightFiller.swift
//  espresso
//
//  Created by Zhihao Li on 4/15/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import GameplayKit // For Gaussian Rand

public typealias WeightFiller = () -> Tensor.DataType

public func gaussianWeightFiller(mean mean: Tensor.DataType, std: Tensor.DataType) -> () -> Tensor.DataType {
  return {() -> Tensor.DataType in
    return Tensor.DataType(0.0)
  }
}