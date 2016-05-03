//
//  SoftmaxLayerTests.swift
//  espresso
//
//  Created by Jerry Zhang on 5/1/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest

class SoftmaxLayerTests: XCTestCase {


  var layer : SoftmaxLayer? = nil
  let params = SoftmaxParameters(name: "Softmax Layer Test",
                                 dependencies: ["Conv Layer"], /* whatever */
                                 axis: 1)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomDimensions = [[5,2,3,4]]

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    layer = SoftmaxLayer(parameters: params)
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
    let channels = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, channels, height, width]]
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, bottomDimensions[0])
    //XCTAssertEqual((layer?.gradient.dimensions)!, outputDimension)
  }

  func testForwardCPU() {
    let batchSize = 1
    let chan = 3
    let height = 4
    let width = 4
    let bottomDimensions = [[batchSize, chan, height, width]]
    let bottom = Tensor(dimensions: [batchSize, chan, height, width])
    bottom.storage = [ 1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 1,

                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,

                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0]
    let network = NetworkProperties(batchSize: 1, engine: .CPU)
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)

    layer?.forwardCPU([bottom])

    let output = layer?.output
    let Z:Float = exp(0) + exp(-1) + exp(-1)
    let expected: [Float] = [
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/Z,

      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, exp(-1)/Z,

      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, exp(-1)/Z
      ]
    XCTAssertEqual(output!.storage, expected)
  }

}
