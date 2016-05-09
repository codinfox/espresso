//
//  Network+Import.swift
//  espresso
//
//  Created by Zhihao Li on 4/30/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

public struct CompressedInfo {
  public var codebook : [Float32]
  public var spm : [UInt8] // sparse matrix value index representation
  public var ind : [Int32] // sparse matrix non-zero element position index
  public init(codebook: [Float32],
              spm: [UInt8],
              ind: [Int32]) {
    self.codebook = codebook
    self.spm = spm
    self.ind = ind
  }
}

extension Network {
  public func importFromFile(filename: String, engine: NetworkProperties.NetworkEngine = .CPU) {
    let data = NSData(contentsOfFile: filename)
    let params = NSKeyedUnarchiver.unarchiveObjectWithData(data!) as! NSDictionary as Dictionary
    for (layerName, param) in params {
      let param = param as! [[Float]]
      let layerIndex = layerNameIndexMapping[layerName as! String]
      let layer = layers[layerIndex!] as! TrainableLayerProtocol

      assert(layer.weights.numel == param[0].count)
      layer.weights.storage = param[0] as! [Tensor.DataType]
      assert(layer.bias.numel == param[1].count)
      layer.bias.storage = param[1] as! [Tensor.DataType]
      if (engine == .GPU) {
        layer.weights.putToMetal()
        layer.bias.putToMetal()
      }

      layer.bias.readyToUse = true
      layer.weights.readyToUse = true
    }
  }

  /* referenced from: https://github.com/songhan/Deep-Compression-AlexNet/blob/master/decode.py

   @article{han2015deep_compression,
   title={Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding},
   author={Han, Song and Mao, Huizi and Dally, William J},
   journal={International Conference on Learning Representations (ICLR)},
   year={2016}
   }
   */
  func restoreWeightsByDecompression(rawdata: NSData, nonZeroElements: Int, bits: Int, biasNum: Int) -> (info: CompressedInfo, bias: [Float32]) {

    var dataPointer : Int = 0

    let codebookSize = 1 << bits
    var codebook = [Float32](count: codebookSize, repeatedValue: 0)
    rawdata.getBytes(&codebook, range: NSRange(location: dataPointer, length: codebookSize * sizeof(Float32)))
    dataPointer += codebookSize * sizeof(Float32)

    // FIXME: Bias should not be restored from time to time, just once
    var bias = [Float32](count: biasNum, repeatedValue: 0)
    rawdata.getBytes(&bias, range: NSRange(location: dataPointer, length: biasNum * sizeof(Float32)))
    dataPointer += biasNum * sizeof(Float32)

    var spmStream = [UInt8](count: ((nonZeroElements-1) / (8/bits) + 1), repeatedValue: 0)
    rawdata.getBytes(&spmStream, range: NSRange(location: dataPointer, length: spmStream.count * sizeof(UInt8)))
    dataPointer += spmStream.count * sizeof(UInt8)

    var indStream = [UInt8](count: ((nonZeroElements-1) / 2 + 1), repeatedValue: 0)
    rawdata.getBytes(&indStream, range: NSRange(location: dataPointer, length: indStream.count * sizeof(UInt8)))
    dataPointer += indStream.count * sizeof(UInt8)

    let slots = (bits == 4) ? 2 : 1

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

    return (info: CompressedInfo(codebook: codebook, spm: spm, ind: ind), bias: bias)
    // Can be SIMD
//    for i in 0 ..< ind.count {
//      codes[Int(ind[i])] = codebook[Int(spm[i])]
//    }
  }

  public func importCompressedNetworkFromFile(filename: String) {
    let data = NSData(contentsOfFile: filename)!
    var dataPointer : Int = 0
    var trainableLayers : [TrainableLayerProtocol] = []
    for layer in self.layers {
      if layer is TrainableLayerProtocol {
        trainableLayers.append(layer as! TrainableLayerProtocol)
      }
    }
    var numNonZeroElements = [UInt32](count: trainableLayers.count, repeatedValue: 0)
    data.getBytes(&numNonZeroElements, range: NSRange(location: dataPointer, length: numNonZeroElements.count * sizeof(UInt32)))
    dataPointer += numNonZeroElements.count * sizeof(UInt32)

    for idx in trainableLayers.indices {
      var layer = trainableLayers[idx]
      var byteCount = 0
      var bits = 0
      if layer is ConvolutionLayer {
        bits = 8
      } else if layer is FullyConnectedLayer {
        bits = 4
      }

      byteCount += (1 << bits) * sizeof(Float32) // codebook size
      byteCount += layer.bias.numel * sizeof(Float32) // bias size
      byteCount += (Int(numNonZeroElements[idx]) - 1)/(8/bits)+1 // spm_stream
      byteCount += (Int(numNonZeroElements[idx]) - 1)/2+1 // ind_stream

      let rawdata : NSData = data.subdataWithRange(NSRange(location: dataPointer, length: byteCount))
      dataPointer += byteCount
      
      let (compressionInfo, biasStorage) = restoreWeightsByDecompression(rawdata, nonZeroElements: Int(numNonZeroElements[idx]), bits: bits, biasNum: layer.bias.numel)

      layer.bias.storage = biasStorage
      layer.compressedInfo = compressionInfo

      layer.weights.purgeStorage()
      layer.weights.readyToUse = true // FIXME
    }

  }
}