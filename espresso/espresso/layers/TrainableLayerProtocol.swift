//
//  TrainableLayerProtocol.swift
//  espresso
//
//  Created by Zhihao Li on 4/15/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

public struct CompressedInfo {
  public var compressedStorage : NSData
  public var nonZeroElements : Int
  public var codeBits : Int
  public init(compressedStorage: NSData,
              nonZeroElements: Int,
              codeBits : Int) {
    self.compressedStorage = compressedStorage
    self.nonZeroElements = nonZeroElements
    self.codeBits = codeBits
  }
}

protocol TrainableLayerProtocol : LayerProtocol {
  var weights : Tensor! { get set }
  var bias : Tensor! { get set }
  var compressedInfo : CompressedInfo! { get set }
  var memoryLimitedMode : Bool { get }

  mutating func initWeights()
  /** Take in the gradient of the weights, and update weights
   The update procedure should be conducted by the solver but, besides the global learning rate and other parameters, also consider the local learning rate and parameters (weight decay and etc.)
   */
  mutating func updateWeights(weightGrad: Tensor) // TODO

  mutating func initBias()

  mutating func updateBias(biasGrad: Tensor)

  // MARK: Decompression

  mutating func purgeWeights()

  func restoreWeightsByDecompression()
}

extension TrainableLayerProtocol {
  var memoryLimitedMode : Bool {
    return self.compressedInfo != nil
  }

  mutating func purgeWeights() {
    self.weights.purgeStorage()
  }

  /* referenced from: https://github.com/songhan/Deep-Compression-AlexNet/blob/master/decode.py

   @article{han2015deep_compression,
      title={Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding},
      author={Han, Song and Mao, Huizi and Dally, William J},
      journal={International Conference on Learning Representations (ICLR)},
      year={2016}
   }
   */
  func restoreWeightsByDecompression() {
    if self.bias.readyToUse && self.weights.readyToUse {
      return
    }

    let rawdata = self.compressedInfo.compressedStorage
    let nonZeroElements = self.compressedInfo.nonZeroElements
    let bits = self.compressedInfo.codeBits

    var dataPointer : Int = 0

    let codebookSize = 1 << bits
    var codebook = [Float32](count: codebookSize, repeatedValue: 0)
    rawdata.getBytes(&codebook, range: NSRange(location: dataPointer, length: codebookSize * sizeof(Float32)))
    dataPointer += codebookSize * sizeof(Float32)

    // FIXME: Bias should not be restored from time to time, just once
    if !self.bias.readyToUse {
      self.bias.storage = [Float32](count: self.bias.numel, repeatedValue: 0)
      rawdata.getBytes(&self.bias.storage, range: NSRange(location: dataPointer, length: self.bias.numel * sizeof(Float32)))
      self.bias.readyToUse = true
    }
    dataPointer += self.bias.numel * sizeof(Float32)

    var spmStream = [UInt8](count: ((nonZeroElements-1) / (8/bits) + 1), repeatedValue: 0)
    rawdata.getBytes(&spmStream, range: NSRange(location: dataPointer, length: spmStream.count * sizeof(UInt8)))
    dataPointer += spmStream.count * sizeof(UInt8)

    var indStream = [UInt8](count: ((nonZeroElements-1) / 2 + 1), repeatedValue: 0)
    rawdata.getBytes(&indStream, range: NSRange(location: dataPointer, length: indStream.count * sizeof(UInt8)))
    dataPointer += indStream.count * sizeof(UInt8)

    let slots = (bits == 4) ? 2 : 1

    var codes = [Float32](count: self.weights.numel, repeatedValue: 0)

    // Recover from binary stream
    var spm = [UInt8](count: nonZeroElements, repeatedValue: 0)
    var ind = [Int32](count: nonZeroElements, repeatedValue: 0)

    if slots == 2 {
      // Can be SIMD
      for i in 0.stride(to: nonZeroElements, by: 2) {
        spm[i] = spmStream[i/2] % 16 // (1 << 4)
        if i + 1 < nonZeroElements {
          spm[i+1] = spmStream[i/2] / 16 // (1 << 4)
        }
      }
    } else { // slots == 1
      spm = spmStream
    }

    // Can be SIMD
    for i in 0.stride(to: nonZeroElements, by: 2) {
      ind[i] = Int32(indStream[i/2] % 16) // (1 << 4)
      if i + 1 < nonZeroElements {
        ind[i+1] = Int32(indStream[i/2] / 16) // (1 << 4)
      }
    }

    // Can be SIMD
    ind[0] += 1
    for i in 1 ..< ind.count {
      ind[i] += (ind[i - 1] + 1)
    }
    // Can be SIMD
    for i in ind.indices {
      ind[i] -= 1
    }

    // Can be SIMD
    for i in 0 ..< ind.count {
      codes[Int(ind[i])] = codebook[Int(spm[i])]
    }
    self.weights.storage = codes
    self.weights.readyToUse = true
  }

  mutating func initWeights() {
    // Do nothing
  }
  mutating func updateWeights(weightGrad: Tensor) {
    // Do nothing
  }
  mutating func initBias() {
    // Do nothing
  }
  mutating func updateBias(biasGrad: Tensor) {
    // Do nothing
  }
}