//
//  MetalParameters.swift
//  espresso
//
//  Created by Jerry Zhang on 4/18/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation


public struct MetalConvolutionParameter {
  var padSize: Int
  var kernelSize: Int
  var stride: Int
  var inputChannel: Int
  var inputHeight: Int
  var inputWidth: Int
  var outputChannel: Int
  var outputHeight: Int
  var outputWidth: Int
}


public struct MetalReluParameter {
  var negativeSlope: Float
  init(negativeSlope: Float, inputDim: [Int]) {
    self.negativeSlope = negativeSlope
  }
}

public struct MetalPoolingParameter {
  var padSize: Int
  var stride: Int
  var inputChannel: Int
  var inputHeight: Int
  var inputWidth: Int
  var outputChannel: Int
  var outputHeight: Int
  var outputWidth: Int
}

public struct MetalFullyConnectedParameter {
  var numNeurons: Int
  var channel: Int
  var height: Int
  var width: Int
}

public struct MetalSoftmaxParameter {
  var height: Int
  var width: Int
}

public struct MetalSoftmaxWithLossParameter {

}

public struct MetalDropoutParameter {

}

public struct MetalLrnParameter {

}
