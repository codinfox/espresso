//
//  Network+Import.swift
//  espresso
//
//  Created by Zhihao Li on 4/30/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

extension Network {
  public func importFromFile(filename: String) {
    let data = NSData(contentsOfFile: filename)
    let params = NSKeyedUnarchiver.unarchiveObjectWithData(data!) as! NSDictionary as Dictionary
    for (layerName, param) in params {
      let layerIndex = layerNameIndexMapping[layerName as! String]
      let layer = layers[layerIndex!] as! TrainableLayerProtocol

      assert(layer.weights.numel == param[0].count)
      layer.weights.storage = param[0] as! [Tensor.DataType]
      assert(layer.bias.numel == param[1].count)
      layer.bias.storage = param[1] as! [Tensor.DataType]
    }
  }
}