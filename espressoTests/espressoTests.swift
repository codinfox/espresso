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
    //let params = ImageDataParameters(imgNames: ["mnist_train.csv"], dimensions: [], readImage: readImage)
    //let imgDataLayer = ImageDataLayer(name:"Image Data Layer Test", parameters:params)
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
    return ([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],[1])
  }

  func testExample() {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    // dimension: channel * height * width
    let dimension = [1, 6, 4]
    let params = ImageDataParameters(imgNames: ["mnist_train.csv"], dimensions: dimension, readImage: readImage)
    let imgDataLayer = ImageDataLayer(name:"Image Data Layer Test", parameters:params)

    var bottomOpt = [Tensor(dimensions: dimension)]
    bottomOpt[0].storage = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]

    imgDataLayer.reshape(dimension)
    let allZeros = Array(count: 24, repeatedValue: 0)
    XCTAssert(imgDataLayer.output[0].storage == allZeros, "ImageDataLayer: storage should be initialized to allZeros!" + imgDataLayer.output[0].storage.debugDescription)
    imgDataLayer.forwardCPU(bottomOpt)
    XCTAssert(imgDataLayer.output[0].storage == bottomOpt[0].storage, "ImageDataLayer: output should be equal to input!" + imgDataLayer.output[0].storage.debugDescription)
  }

}


class ConvolutionLayerTest0: XCTestCase {

  func testExample() {

  }
}
