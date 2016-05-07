//
//  SqueezeNetTests.swift
//  espresso
//
//  Created by Zhihao Li on 5/2/16.
//  Copyright © 2016 CMU. All rights reserved.
//


import XCTest
@testable import espresso

class AlexNetTests: XCTestCase {

  var network : Network!
  let filename = "/Users/Ben/Projects/espresso/models/AlexNet_compressed.net"

  func readUIImageToTensor() -> Tensor {
    let inputCGImage = UIImage(contentsOfFile: "/Users/Ben/Downloads/ING-bell-pepper_sql.jpg")!.CGImage
    let width = 227 // CGImageGetWidth(inputCGImage)
    let height = 227 // CGImageGetHeight(inputCGImage)

    let tensor = Tensor(dimensions: [1,3,height,width]) // FIXME: demo

    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let bitsPerComponent = 8

    let pixels = UnsafeMutablePointer<UInt32>(calloc(height * width, sizeof(UInt32)))
    let colorSpace = CGColorSpaceCreateDeviceRGB()

    let context = CGBitmapContextCreate(pixels, width, height, bitsPerComponent, bytesPerRow, colorSpace, CGImageAlphaInfo.PremultipliedFirst.rawValue)

    // will automatically resize image to [width, height]
    CGContextDrawImage(context, CGRectMake(0, 0, CGFloat(width), CGFloat(height)), inputCGImage)

    let dataPointer = UnsafePointer<UInt8>(pixels)

    for j in 0 ..< height {
      for i in 0 ..< width {
        let offset = 4*((Int(width) * Int(j)) + Int(i))
        //        let alphaValue = dataType[offset] as UInt8
        tensor[0,0,j,i] = Tensor.DataType(dataPointer[offset+3]) - 104 // blue (- mean)
        tensor[0,1,j,i] = Tensor.DataType(dataPointer[offset+2]) - 117 // green (- mean)
        tensor[0,2,j,i] = Tensor.DataType(dataPointer[offset+1]) - 123 // red (- mean)
      }
    }

    free(pixels)
    return tensor
  }

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
    network = Network(parameters: NetworkProperties(batchSize: 1, engine: .CPU))
    network.add(ImageDataLayer(parameters: ImageDataParameters(
      name: "data",
      imgNames: Array<String>(count: 100, repeatedValue: ""),
      dimensions: [1,3,227,227],
      dependencies: [],
      readImage: { _ in (self.readUIImageToTensor().storage, [0])}
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv1",
      dependencies: ["data"],
      numOutput: 96,
      kernelSize: 11,
      stride: 4
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv1",
      dependencies: ["conv1"]
      )))

    // LRN layer here

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool1",
      dependencies: ["relu_conv1"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv2",
      dependencies: ["pool1"],
      numOutput: 256,
      kernelSize: 5,
      padSize: 2
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv2",
      dependencies: ["conv2"]
      )))

    // LRN layer here

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool2",
      dependencies: ["relu_conv2"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv3",
      dependencies: ["pool2"],
      numOutput: 384,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv3",
      dependencies: ["conv3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv4",
      dependencies: ["relu_conv3"],
      numOutput: 384,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv4",
      dependencies: ["conv4"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv5",
      dependencies: ["relu_conv4"],
      numOutput: 256,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv5",
      dependencies: ["conv5"]
      )))

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool5",
      dependencies: ["relu_conv5"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(FullyConnectedLayer(parameters: FullyConnectedParameters(
      name: "fc6",
      dependencies: ["pool5"],
      numOutput: 4096
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu6",
      dependencies: ["fc6"]
      )))
    network.add(FullyConnectedLayer(parameters: FullyConnectedParameters(
      name: "fc7",
      dependencies: ["relu6"],
      numOutput: 4096
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu7",
      dependencies: ["fc7"]
      )))
    network.add(FullyConnectedLayer(parameters: FullyConnectedParameters(
      name: "fc8",
      dependencies: ["relu7"],
      numOutput: 1000
      )))
    network.add(SoftmaxLayer(parameters: SoftmaxParameters(
      name: "prob",
      dependencies: ["fc8"]
      )))
  }

  override func tearDown() {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testPerformanceExample() {
    // This is an example of a performance test case.
    self.network.importCompressedNetworkFromFile(filename)
    self.measureBlock {
      self.network.forward()
    }
    let out = (network.layers.last as! ForwardLayerProtocol).output.storage
    let prob = out.maxElement()
    let index = out.indexOf(prob!)
    XCTAssertEqual(index!, 945)
  }
}

