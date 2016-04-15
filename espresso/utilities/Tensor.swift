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
public class Tensor<DataType> {
  var storage : [DataType] = []
  
  public private(set) var dimensions : [Int] = []
  public private(set) var numel : Int = 0
  private var indexAuxilary: [Int] = [1]
  /**
   * Initialize the Tensor with dimensionalities
   */
  init(dimensions: [Int]) {
    self.dimensions = dimensions
    self.numel = 1
    for d in dimensions.reverse() {
      self.numel *= d
      indexAuxilary.append(self.numel)
    }
    indexAuxilary.removeLast()
    indexAuxilary = indexAuxilary.reverse()

    self.storage.reserveCapacity(numel)
  }
  
  func index(idxs: [Int]) -> Int {
    var idx = 0
    for i in 0..<indexAuxilary.count {
      idx += indexAuxilary[i] * idxs[i]
    }
    return idx
  }
  
  subscript(idxs: Int...)->DataType {
    get {
      return self.storage[index(idxs)]
    }

    set(newValue) {
      self.storage[index(idxs)] = newValue
    }
  }
  
}