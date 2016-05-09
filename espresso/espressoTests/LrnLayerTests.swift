//
//  LrnLayerTests.swift
//  espresso
//
//  Created by Jerry Zhang on 5/9/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest

class LrnLayerTests: XCTestCase {

  var layer : LRNLayer! = nil
  //let params = LRNParameters(name: "lrn", dependencies: [], localSize: 1, alpha: 0.0001, beta: 5, region: LRNParameters.)
  let network = NetworkProperties(batchSize: 5, engine: .CPU)
  let bottomDimensions = [[5,2,3,4]]

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
  }

  override func tearDown() {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testExample() {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
  }

  func testPerformanceExample() {
    // This is an example of a performance test case.
    self.measureBlock {
      // Put the code you want to measure the time of here.
    }
  }

}
