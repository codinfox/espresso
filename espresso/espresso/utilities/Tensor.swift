//
//  Tensor.swift
//  espresso
//
//  Created by Zhihao Li on 4/12/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief Basic storage class
 *  Tensor is a multidimensional matrix. This serves as the fundamental storage class.
 *  Tensor can take arbitrary type of data and when using, should be initialized with the dimension.
 */
public class Tensor {
  public typealias DataType = Float

  public var engine: NetworkProperties.NetworkEngine {
    get {
      if self.metalDevice == nil {
        return .CPU
      } else {
        return .GPU
      }
    }
  }
  public var storage : [DataType] = []
  public var mtlStorage : MTLBuffer!
  public var metalDevice: MTLDevice!

  public private(set) var dimensions : [Int] = []
  public private(set) var numel : Int = 0
  public var capacity : Int = 0
  var indexAuxilary: [Int] = []
  var readyToUse : Bool = false

  /**
   * Initialize the Tensor with dimensionalities
   */
  public init(metalDevice: MTLDevice!, dimensions:[Int]=[]) {
    self.metalDevice = metalDevice
    self.dimensions = dimensions
  }

  public init(dimensions: [Int]) {
    self.reshape(dimensions)
    self.readyToUse = true
  }

  func index(idxs: [Int]) -> Int {
    var idx = 0
    for i in idxs.indices {
      idx += indexAuxilary[i] * idxs[i]
    }
    return idx
  }

  public func count(fromDimension fromDimension: Int = 0, toDimension: Int = -1) -> Int {
    guard fromDimension < dimensions.count else {
      return 1
    }
    var toDimension = toDimension
    if toDimension < 0 {
      toDimension = dimensions.count + toDimension
    }
    assert(fromDimension <= toDimension)

    return self.dimensions[fromDimension...toDimension].reduce(1, combine: {$0 * $1})
  }

  /* Put content in storage to mtl */
  public func putToMetal() {
    self.mtlStorage = createFloatArray(self.storage, metalDevice: self.metalDevice)
  }

  /* Get the result in mtl */
  public func getFromMetal() {
    if (mtlStorage != nil) {
      let length = count() * sizeof(Float)
      self.storage = [Float](count: count(), repeatedValue: 0)
      let mtlContent = NSData(bytesNoCopy: self.mtlStorage.contents(),
                              length: length, freeWhenDone: false)
      mtlContent.getBytes(&self.storage, length:length)
    }
  }

  public func reshape(dimensions: [Int]) {
    switch engine {
    case .CPU:
      reshapeCPU(dimensions)
    case .GPU:
      reshapeGPU(dimensions)
    }
  }

  public func reshapeGPU(dimensions: [Int]) {
    if self.dimensions == dimensions {
      if (self.mtlStorage == nil) {
        self.mtlStorage = metalDevice.newBufferWithLength(self.capacity * sizeof(DataType), options: MTLResourceOptions.CPUCacheModeDefaultCache)
      }
      return
    }
    let numNewElements = dimensions.count == 0 ? 0 : dimensions.reduce(1, combine: {$0 * $1})
    if self.capacity < numNewElements {
      self.mtlStorage = metalDevice.newBufferWithLength(numNewElements * sizeof(DataType), options: MTLResourceOptions.CPUCacheModeDefaultCache)
      self.capacity = numNewElements
    }
    self.dimensions = dimensions
    self.numel = numNewElements
    assert(self.numel >= 0)
    // delete later after debug
    if self.numel != 0 {
      self.indexAuxilary = [1]
      for d in dimensions.reverse() {
        indexAuxilary.append(d * indexAuxilary.last!)
      }
      assert(indexAuxilary.last! == self.numel, "number of elements in Tensor doesn't match")
      indexAuxilary.removeLast()
      indexAuxilary = indexAuxilary.reverse()
    } else {
      // FIXME: bad hack?
      self.indexAuxilary = []
    }
  }

  public func reshapeCPU(dimensions: [Int]) {
    if self.dimensions == dimensions {
      return
    }
    let numNewElements = dimensions.count == 0 ? 0 : dimensions.reduce(1, combine: {$0 * $1})
    if self.capacity < numNewElements {
      self.storage = Array(count: numNewElements, repeatedValue: 0)
      self.capacity = numNewElements
    }
    self.dimensions = dimensions
    self.numel = numNewElements
    assert(self.numel >= 0)

    if self.numel != 0 {
      self.indexAuxilary = [1]
      for d in dimensions.reverse() {
        indexAuxilary.append(d * indexAuxilary.last!)
      }
      assert(indexAuxilary.last! == self.numel, "number of elements in Tensor doesn't match")
      indexAuxilary.removeLast()
      indexAuxilary = indexAuxilary.reverse()
    } else {
      // FIXME: bad hack?
      self.indexAuxilary = []
    }
  }

  public func reset(val: DataType) {
    for i in 0 ..< self.numel {
      self.storage[i] = val
    }
  }

  public subscript(idxs: Int...)->DataType {
    get {
      // May be exceptions
      return self.storage[index(idxs)]
    }

    set(newValue) {
      self.storage[index(idxs)] = newValue
    }
  }

  // MARK: Decompression

  /**
   When memory limited, storage may be purged to save space
   */
  public func purgeStorage() {
    self.storage = []
    readyToUse = false
  }
}