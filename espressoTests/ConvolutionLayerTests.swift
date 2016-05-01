//
//  ConvolutionLayerTests.swift
//  espresso
//
//  Created by Zhihao Li on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
@testable import espresso

class ConvolutionLayerFeedforwardTests: XCTestCase {

  var layer : ConvolutionLayer? = nil
  let params = ConvolutionParameters(name: "Conv Layer Test",
                                     dependencies: ["Image Data Layer"],
                                     numOutput: 2,
                                     kernelSize: 2,
                                     stride: 2,
                                     padSize: 2,
                                     isBiasTerm: true)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomDimensions = [[5,2,3,4]]
  // assert network.batchSize == bottomDimens[0][0]??

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    layer = ConvolutionLayer(parameters: params)
  }

  override func tearDown() {
    layer = nil
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testLayerSetUp() {
    let channels = params.numOutput
    let kernelSize = params.kernelSize
    let bottomChannels = bottomDimensions[0][1]
    let weightDim = [channels, bottomChannels, kernelSize, kernelSize]
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.weights.dimensions)!, weightDim)
    XCTAssertEqual((layer?.bias.dimensions)!, [channels])
    XCTAssertEqual(layer?.output.dimensions[0], network.batchSize)
    XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  }

  func testReshape() {
    let batchSize = 1
    let bottomNumOutput = 3
    //let chan = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    let outHeight = (height + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    let outWidth = (width + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, [batchSize, params.numOutput, outHeight, outWidth])
    XCTAssertEqual((layer?.gradient.dimensions)!, [batchSize, params.numOutput, outHeight, outWidth])
  }

  func testForwardCPU() {
    let batchSize = 1
    let chan = 3
    let height = 4
    let width = 4
    let bottomDimensions = [[batchSize, chan, height, width]]
    let bottom = Tensor(dimensions: [batchSize, chan, height, width])
    bottom.storage = [ 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                      13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                      13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                      13,14,15,16]
    let network = NetworkProperties(batchSize: 1, engine: .CPU)
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)

    layer?.weights.storage = [1,1,
                              1,1,
                              1,1,
                              1,1,
                              0,0,
                              0,0,

                               2, 2,
                               2, 2,
                              -1,-1,
                              -1,-1,
                               1, 1,
                               1, 1]
    layer?.bias.storage = [100,200]

    layer?.forwardCPU([bottom])

    let output = layer?.output
    XCTAssertEqual(output!.storage, [
      100,100,100,100,
      100,128,144,100,
      100,192,208,100,
      100,100,100,100,

      200,200,200,200,
      200,228,244,200,
      200,292,308,200,
      200,200,200,200
      ])

  }
}
