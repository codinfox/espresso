//
//  MnistTests.swift
//  espresso
//
//  Created by Jerry Zhang on 5/1/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest

class MnistTests: XCTestCase {

  var network : Network? = nil
  let mnistTrainPath = "/Users/jerry/Projects/15-618/mnist_train_toy.csv"
  let mnistTestPath = "/Users/jerry/Projects/15-618/mnist_test_toy.csv"
  let modelFileName = "/Users/jerry/Projects/15-618/espresso/models/lenet.espressomodel"
  var curImageNo = 0
  var trainingData:[[Tensor.DataType]] = []
  var trainingLabels:[Tensor.DataType] = []
  let imageNum = 10

  func initImages(name: String) {
    let csvText:String?
    do  {
      csvText = try String(contentsOfFile: mnistTrainPath, encoding:NSUTF8StringEncoding)
    } catch _ {
      csvText = nil
    }
    let csv = CSwiftV(String: csvText!)
    let rows = csv.rows
    let cols:Int = csv.columnCount
    let intRows = rows.map(
      {(x:[String]) -> [Tensor.DataType] in return x.map({(y:String)->Tensor.DataType in return Tensor.DataType(y)!})})
    trainingLabels = intRows.map({(x:[Tensor.DataType])->Tensor.DataType in return x[0]})
    trainingData = intRows.map({(x:[Tensor.DataType])->[Tensor.DataType] in return Array(x[1..<cols])})
  }

  func readImage(name: String) -> ([Float], [Float]) {
    let data = trainingData[curImageNo]
    let label = trainingLabels[curImageNo]
    curImageNo+=1
    return (data, [label])
  }

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    network = Network(parameters: NetworkProperties(batchSize: 1, engine: .CPU))
    network?.add(ImageDataLayer(parameters: ImageDataParameters(
      name: "data",
      imgNames: Array(count: imageNum, repeatedValue: ""),
      dimensions: [1,1,28,28],
      dependencies: [],
      readImage: { _ in self.readImage("") }
      )))
    network?.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv1",
      dependencies: ["data"],
      numOutput: 20,
      kernelSize: 5,
      isBiasTerm: true
      )))
    network?.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool1",
      dependencies: ["conv1"],
      kernelSize: 2,
      stride: 2,
      method: .MAX
      )))
    network?.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv2",
      dependencies: ["pool1"],
      numOutput: 50,
      kernelSize: 5,
      isBiasTerm: true
      )))
    network?.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool2",
      dependencies: ["conv2"],
      kernelSize: 2,
      stride: 2,
      method: .MAX
      )))
    network?.add(FullyConnectedLayer(parameters: FullyConnectedParameters(
      name: "ip1",
      dependencies: ["pool2"],
      numOutput: 500,
      isBiasTerm: true
      )))
    network?.add(ReluLayer(parameters: ReLUParameters(
      name: "relu1",
      dependencies: ["ip1"]
      )))
    network?.add(FullyConnectedLayer(parameters: FullyConnectedParameters(
      name: "ip2",
      dependencies: ["relu1"],
      numOutput: 10,
      isBiasTerm: true
      )))
    network?.add(SoftmaxLayer(parameters: SoftmaxParameters(
      name: "prob",
      dependencies: ["ip2"]
      )))
  }

  override func tearDown() {
    network = nil
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testImport() {
    if let network = network {
      initImages(mnistTrainPath)
      network.importFromFile(modelFileName)
      for _ in 0..<imageNum {
        network.forward()
      }
    }
  }

}
