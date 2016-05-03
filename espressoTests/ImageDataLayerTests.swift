//
//  espressoTests.swift
//  espressoTests
//
//  Created by Zhihao Li on 4/11/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
@testable import espresso

/* TODO: Testing reading images from csv file(MNIST) */
class ImageDataLayerTest0: XCTestCase {

  /* name -> (train, test) */
  func readImage(name: String) -> ([Float], [Float]) {
    let trainFile = name
    //let testFile = "mnist_test.csv"

    var data:String?
    if let dir : NSString = NSSearchPathForDirectoriesInDomains(NSSearchPathDirectory.DocumentDirectory, NSSearchPathDomainMask.AllDomainsMask, true).first {
      let path = dir.stringByAppendingPathComponent(trainFile);
      //reading
      do {
        data = try String(contentsOfFile: path, encoding: NSUTF8StringEncoding)
      }
      catch {/* error handling here */}
    }
    let csv = CSwiftV(String: data!)
    let rows = csv.rows
    let cols:Int = csv.columnCount
    let intRows = rows.map(
      {(x:[String]) -> [Float] in return x.map({(y:String)->Float in return Float(y)!})})
    let trainingLabels = intRows.map({(x:[Float])->Float in return x[0]})
    let trainingData = intRows.flatMap({(x:[Float])->[Float] in return Array(x[1..<cols])})
    return (Array(trainingData[0..<784]), [trainingLabels[0]])
  }

  func testExample() {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    let dim = [1, 1, 28, 28]
    let params = ImageDataParameters(name: "Image Layer Test", imgNames: ["small.csv"], dimensions: dim, dependencies: [], readImage: readImage)
    let imgDataLayer = ImageDataLayer(parameters:params)
    imgDataLayer.layerSetUp(engine: .CPU, bottomDimensions: [dim])
    imgDataLayer.reshapeByBottomDimensions([dim])
    imgDataLayer.forwardCPU([])
  }

}

class ImageDataLayerTest1: XCTestCase {

  func readImage(name: String) -> ([Float], [Float]) {
    return ([1,2,3,4,
             1,2,3,4,
             1,2,3,4,
             1,2,3,4,
             1,2,3,4,
             1,2,3,4],[1])
  }

  func testExample() {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    // dimension: batchSize * channel * height * width
    let dimension = [1, 1, 6, 4]
    let params = ImageDataParameters(name: "Image Layer Test", imgNames: ["mnist_train.csv"], dimensions: dimension, dependencies: [], readImage: readImage)
    let imgDataLayer = ImageDataLayer(parameters:params)

    let bottomOpt = Tensor(dimensions: dimension)
    bottomOpt.storage = [1,2,3,4,
                         1,2,3,4,
                         1,2,3,4,
                         1,2,3,4,
                         1,2,3,4,
                         1,2,3,4]
    imgDataLayer.layerSetUp(engine: .CPU, bottomDimensions: [dimension])
    imgDataLayer.reshapeByBottomDimensions([dimension])
    let allZeros = Array(count: bottomOpt.count(), repeatedValue: 0)
    XCTAssert(imgDataLayer.output.storage == allZeros, "ImageDataLayer: storage should be initialized to allZeros!" + imgDataLayer.output.storage.debugDescription)
    imgDataLayer.forwardCPU([])
    XCTAssert(imgDataLayer.output.storage == bottomOpt.storage, "ImageDataLayer: output should be equal to input!" + bottomOpt.storage.debugDescription)
  }

}