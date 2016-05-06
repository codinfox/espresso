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

  var layer : ReluLayer? = nil
  let params = ReLUParameters(name: "Relu Layer Test",
                              dependencies: ["Conv Layer"],
                              negativeSlope: 2)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomDimensions = [[5,2,3,4]]

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    layer = ReluLayer(parameters: params)
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
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, bottomDimensions[0])
    //XCTAssertEqual((layer?.gradient.dimensions)!, bottomDimensions[0])
  }

  func testForwardCPU() {
    let batchSize = 2
    let chan = 3
    let height = 4
    let width = 4
    let bottomDimensions = [[batchSize, chan, height, width]]
    let bottom = Tensor(dimensions: [batchSize, chan, height, width])
    bottom.storage = [  1, -2, 3, 4, /* channel 1 */
      5, 6, -7, 8,
      9,-10,11,-12,
      13,14,-15,16,

      -1, 2, 3, 4,
      5, -6, 7, -8,
      9,10,-11,12,
      13,-14,15,16,

      1, 2, 3, -4,
      5, 6, -7, 8,
      9,-10,11,12,
      13,14,-15,16,

      1, -2, 3, 4, /* channel 2 */
      5, 6, -7, 8,
      9,-10,11,-12,
      13,14,-15,16,

      -1, 2, 3, 4,
      5, -6, 7, -8,
      9,10,-11,12,
      13,-14,15,16,

      1, 2, 3, -4,
      5, 6, -7, -8,
      9,-10,11,12,
      13,14,-15,16]
    let network = NetworkProperties(batchSize: 2, engine: .CPU)
    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    layer?.forwardCPU([bottom])

    let output = layer?.output
    let expected:[Float] = [
      1.0, -4.0, 3.0, 4.0, /* channel 1 */
      5.0, 6.0, -14.0, 8.0,
      9.0,-20.0,11.0,-24.0,
      13.0,14.0,-30.0,16.0,

      -2.0, 2.0, 3.0, 4.0,
      5.0, -12.0, 7.0, -16.0,
      9.0,10.0,-22.0,12.0,
      13.0,-28.0,15.0,16.0,

      1.0, 2.0, 3.0, -8.0,
      5.0, 6.0, -14.0, 8.0,
      9.0,-20.0,11.0,12.0,
      13.0,14.0,-30.0,16.0,

      1.0, -4.0, 3.0, 4.0, /* channel 2 */
      5.0, 6.0, -14.0, 8.0,
      9.0,-20.0,11.0,-24.0,
      13.0,14.0,-30.0,16.0,

      -2.0, 2.0, 3.0, 4.0,
      5.0, -12.0, 7.0, -16.0,
      9.0,10.0,-22.0,12.0,
      13.0,-28.0,15.0,16.0,

      1.0, 2.0, 3.0, -8.0,
      5.0, 6.0, -14.0,-16.0,
      9.0,-20.0,11.0,12.0,
      13.0,14.0,-30.0,16.0]
    XCTAssertEqual(output!.storage, expected)
  }
}
