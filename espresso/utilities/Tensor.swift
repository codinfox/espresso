//
//  Tensor.swift
//  espresso
//
//  Created by Zhihao Li on 4/12/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Basic storage class
 *  Tensor is a multidimensional matrix. This serves as the fundamental storage class.
 *  Tensor can take arbitrary type of data and when using, should be initialized with the dimension.
 */
public class Tensor<StorageDataType> {
  var storage : [StorageDataType] = []

  public private(set) var dimensions : [Int] = []
  public private(set) var numel : Int = 0

  /**
   Initialize the Tensor with dimensionalities
   */
  init(dimensions: [Int]) {
    self.dimensions = dimensions
    self.numel = 1
    for d in dimensions {
      self.numel *= d
    }
    self.storage.reserveCapacity(numel)
  }
}