//
//  FullyConnectedLayerTests.swift
//  espresso
//
//  Created by Jerry Zhang on 5/1/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest

class FullyConnectedLayerTests: XCTestCase {

  var layer : FullyConnectedLayer? = nil
  let params = FullyConnectedParameters(name: "FollyConnected Layer Test",
                                        dependencies: ["Conv Layer"],
                                        numOutput: 2,
                                        isBiasTerm: false,
                                        biasLRMultiplier: 0.5,
                                        weightLRMultiplier: 0.5,
                                        weightFiller: gaussianWeightFiller(mean: 0, std: 1),
                                        biasFiller: gaussianWeightFiller(mean: 0, std: 1))
  let network = NetworkProperties(batchSize: 2, engine: .CPU)
  let bottomDimensions = [[2,1,3,4]]

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    layer = FullyConnectedLayer(parameters: params)
  }

  override func tearDown() {
    layer = nil
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testLayerSetUp() {
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual(layer?.output.dimensions[0], network.batchSize)
    XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  }

  func testReshape() {
    let batchSize = 5
    let bottomNumOutput = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    let outputDimension = [params.numOutput]
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, [batchSize, outputDimension[0]])
    //XCTAssertEqual((layer?.gradient.dimensions)!, outputDimensions)
  }

  func testForwardCPU() {
    let batchSize = 1
    let chan = 3
    let height = 4
    let width = 4
    let bottomDimensions = [[batchSize, chan, height, width]]
    let bottom = Tensor(dimensions: [batchSize, chan, height, width])
    bottom.storage =
      [ 1, 2, 3, 4, /* batch 0, channel 0 */
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16,

        1, 2, 3, 4, /* batch 0, channel 1 */
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16,

        1, 2, 3, 4, /* batch 0, channel 2 */
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16,

        1, 2, 3, 4,  /* batch 1, channel 0 */
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16,

        1, 2, 3, 4,  /* batch 1, channel 1 */
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16,

        1, 2, 3, 4, /* batch 1, channel 2 */
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16]

    layer?.weights.storage =
    [ 1, 2, 3, 4, /* output 0, channel 0 */
      5, 6, 7, 8,
      9,10,11,12,
      13,14,15,16,

      1, 2, 3, 4, /* output 0, channel 1 */
      5, 6, 7, 8,
      9,10,11,12,
      13,14,15,16,

      1, 2, 3, 4, /* output 0, channel 2 */
      5, 6, 7, 8,
      9,10,11,12,
      13,14,15,16,

      1, -1, 1, -1, /* output 1, channel 0 */
      1, -1, 1, -1,
      1, -1, 1, -1,
      1, -1, 1, -1,

      1, -1, 1, -1, /* output 1, channel 1 */
      1, -1, 1, -1,
      1, -1, 1, -1,
      1, -1, 1, -1,

      1, -1, 1, -1, /* output 1, channel 2 */
      1, -1, 1, -1,
      1, -1, 1, -1,
      1, -1, 1, -1]


    let network = NetworkProperties(batchSize: 2, engine: .CPU)
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.forwardCPU([bottom])

    let output = layer?.output
    let expected:[Float] = [4488.0, -24.0]
    XCTAssertEqual(output!.storage, expected)
  }

}
