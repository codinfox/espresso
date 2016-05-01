//
//  PoolingLayerTests.swift
//  espresso
//
//  Created by Jerry Zhang on 5/1/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest

class PoolingLayerTests: XCTestCase {

  var layer : PoolingLayer? = nil
  let params = PoolingParameters(name: "Pooling Layer Test",
                              dependencies: ["Conv Layer"], /* whatever */
                              kernelSize: 3,
                              stride: 2,
                              padSize: 2,
                              method: .MAX)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomDimensions = [[5,2,3,4]]

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    layer = PoolingLayer(parameters: params)
  }

  override func tearDown() {
    layer = nil
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testLayerSetUp() {
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual(layer?.output.dimensions[0], network.batchSize)
    //XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  }

  func testReshape() {
    let batchSize = 3
    let bottomNumOutput = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    let outHeight = (height + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    let outWidth = (width + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, [batchSize, bottomNumOutput, outHeight, outWidth])
    //XCTAssertEqual((layer?.gradient.dimensions)!, [batchSize, bottomNumOutput, outHeight, outWidth])
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

    layer?.forwardCPU([bottom])

    let output = layer?.output
    let expected: [Float] = [
      1,3,4,
      9,11,12,
      13,15,16,

      1,3,4,
      9,11,12,
      13,15,16,

      1,3,4,
      9,11,12,
      13,15,16,
      ]
    XCTAssertEqual(output!.storage, expected)
  }
}
