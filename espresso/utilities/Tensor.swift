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
public class Tensor {
  public typealias DataType = Float

  public var storage : [DataType] = []

  public private(set) var dimensions : [Int] = []
  public private(set) var numel : Int
  public private(set) var capacity : Int = 0
  private var indexAuxilary: [Int] = [1]

  /**
   * Initialize the Tensor with dimensionalities
   */
  init() {
    self.numel = 1
  }
  init(dimensions: [Int]) {
    self.dimensions = dimensions
    self.numel = 1
    for d in dimensions.reverse() {
      self.numel = self.numel * d
      print("n:" + self.numel.description)
      indexAuxilary.append(self.numel)
    }
    indexAuxilary.removeLast()
    indexAuxilary = indexAuxilary.reverse()
    print("numel: " + self.numel.description)
    self.storage.reserveCapacity(numel)
    self.storage = Array(count: self.numel, repeatedValue: 0)
    capacity = self.numel
  }

  func index(idxs: [Int]) -> Int {
    var idx = 0
    for i in 0..<indexAuxilary.count {
      idx += indexAuxilary[i] * idxs[i]
    }
    return idx
  }

  func numElements(dim: [Int]) -> Int {
    return dim.reduce(1, combine: {$0 * $1})
  }

  func reshape(dimensions: [Int]) {
    print("numElements:" + numElements(self.dimensions).description + "numElements: " + numElements(dimensions).description)
    if numElements(self.dimensions) < numElements(dimensions) {
      self.storage = Array(count: numElements(dimensions), repeatedValue: 0)
    }
    self.dimensions = dimensions
  }

  func reset(val: DataType) {
    for i in 0..<dimensions[0] { /* channel */
      for j in 0..<dimensions[1] { /* height */
        for k in 0..<dimensions[2] { /* width */
          storage[index([i, j, k])] = val
        }
      }
    }
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