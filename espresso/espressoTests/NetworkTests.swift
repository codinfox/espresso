//
//  NetworkTests.swift
//  espresso
//
//  Created by Zhihao Li on 4/30/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
@testable import espresso

class NetworkTests: XCTestCase {

  var network : Network? = nil
  let filename = "/Users/Ben/Projects/espresso/models/lenet.espressomodel"

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    network = Network(parameters: NetworkProperties(batchSize: 1, engine: .CPU))
    network?.add(ImageDataLayer(parameters: ImageDataParameters(
      name: "data",
      imgNames: [""],
      dimensions: [1,1,28,28],
      dependencies: [],
      readImage: { _ in ([],[]) }
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
      network.importFromFile(filename)
    }
  }
}