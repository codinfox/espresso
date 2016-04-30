//
//  ReluLayerTests.swift
//  espresso
//
//  Created by Jerry Zhang on 4/30/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
@testable import espresso


class ReluLayerTests: XCTestCase {

  var layer : ConvolutionLayer? = nil
  let params = ConvolutionParameters(name: "Conv Layer Test", dependencies: ["Image Data Layer"], numOutput: 2,
                                     kernelSize: 2,
                                     stride: 2,
                                     padSize: 2,
                                     isBiasTerm: true)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomDimensions = [[1,2,3,4]]

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
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.weights.dimensions)!, bottomDimensions)
    XCTAssertEqual((layer?.bias.dimensions)!, [bottomDimensions[0]])
    XCTAssertEqual(layer?.output.count(), network.batchSize)
    XCTAssertEqual(layer?.gradient.count(), network.batchSize)
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
    XCTAssertEqual((layer?.output.dimensions)!, [params.numOutput, outHeight, outWidth])
    XCTAssertEqual((layer?.gradient.dimensions)!, [params.numOutput, outHeight, outWidth])
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
