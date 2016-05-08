//
//  Network+Import.swift
//  espresso
//
//  Created by Zhihao Li on 4/30/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

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
      
      layer.compressedInfo = CompressedInfo(compressedStorage: rawdata, nonZeroElements: Int(numNonZeroElements[idx]), codeBits: bits)

      layer.bias.purgeStorage()
      layer.weights.purgeStorage()
    }

  }
}