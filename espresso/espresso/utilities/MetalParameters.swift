//
//  MetalParameters.swift
//  espresso
//
//  Created by Jerry Zhang on 4/18/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation


public struct MetalConvolutionParameter {
  var count: Int32
  var padSize: Int32
  var kernelSize: Int32
  var stride: Int32
  var inputChannel: Int32
  var inputHeight: Int32
  var inputWidth: Int32
  var outputChannel: Int32
  var outputHeight: Int32
  var outputWidth: Int32
}


public struct MetalReluParameter {
  var count: Int32
  var negativeSlope: Float
}

public struct MetalPoolingParameter {
  var count: Int32
  var padSize: Int32
  var kernelSize: Int32
  var stride: Int32
  var inputChannel: Int32
  var inputHeight: Int32
  var inputWidth: Int32
  var outputChannel: Int32
  var outputHeight: Int32
  var outputWidth: Int32
}

public struct MetalFullyConnectedParameter {
  var count: Int32
  var numOutput: Int32
  var numElementsPerBatch: Int32
}

public struct MetalSoftmaxParameter {
  var count: Int32
  var numOutput: Int32
  var mapSizeToPerformOn: Int32
}

public struct MetalSoftmaxWithLossParameter {

}

public struct MetalDropoutParameter {

}

public struct MetalLrnParameter {

}
