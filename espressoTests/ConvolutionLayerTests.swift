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
  let params = ConvolutionParameters(numOutput: 2,
                                     kernelSize: 2,
                                     stride: 2,
                                     padSize: 2,
                                     isBiasTerm: true)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomNumOutput = 3

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
    layer?.layerSetUp(network, bottomNumOutput: bottomNumOutput)
    XCTAssertEqual((layer?.weights.dimensions)!, [params.numOutput,bottomNumOutput,params.kernelSize,params.kernelSize])
    XCTAssertEqual((layer?.bias.dimensions)!, [params.numOutput])
    XCTAssertEqual(layer?.output.count, network.batchSize)
    XCTAssertEqual(layer?.gradient.count, network.batchSize)
  }

  func testReshape() {
    let bottomNumOutput = 3
    let chan = 3
    let height = 12
    let width = 20
    let outHeight = (height + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    let outWidth = (width + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    layer?.layerSetUp(network, bottomNumOutput: bottomNumOutput)
    layer?.reshape([chan,height,width])
    XCTAssertEqual((layer?.output[1].dimensions)!, [params.numOutput, outHeight, outWidth])
    XCTAssertEqual((layer?.gradient[1].dimensions)!, [params.numOutput, outHeight, outWidth])
  }

  func testForwardCPU() {
    let chan = 3
    let height = 4
    let width = 4
    let bottom = Tensor(dimensions: [chan, height, width])
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
    layer?.layerSetUp(network, bottomNumOutput: chan)
    layer?.reshape([chan,height,width])

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
    XCTAssertEqual(output![0].storage, [
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
