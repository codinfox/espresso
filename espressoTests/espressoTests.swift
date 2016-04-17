//
//  espressoTests.swift
//  espressoTests
//
//  Created by Zhihao Li on 4/11/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
@testable import espresso

class espressoTests: XCTestCase {
    
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

/* TODO: Testing reading images from csv file(MNIST) */
class ImageDataLayerTest0: XCTestCase {

  /* name -> (train, test) */
  func readImage(name: String) -> ([Float], [Float]) {
    return ([],[])
  }

  func testInit() {
    let params = ImageDataParameters(imgNames: ["mnist_train.csv"], dimensions: [], readImage: readImage)
    let imgDataLayer = ImageDataLayer(name:"Image Data Layer Test", parameters:params)
  }

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

class ImageDataLayerTest1: XCTestCase {

  func readImage(name: String) -> ([Float], [Float]) {
    return ([],[])
  }

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    let params = ImageDataParameters(imgNames: ["mnist_train.csv"], dimensions: [], readImage: readImage)
    let imgDataLayer = ImageDataLayer(name:"Image Data Layer Test", parameters:params)
    // dimension: channel * height * width
    var bottomOpt = [Tensor(dimensions: [1, 6, 4])]
    bottomOpt[0].storage = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
    imgDataLayer.forwardCPU(bottomOpt)
    XCTAssert(imgDataLayer.output[0] === bottomOpt[0], "ImageDataLayer: output should be equal to input!")
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
