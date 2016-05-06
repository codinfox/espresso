//
//  SqueezeNetTests.swift
//  espresso
//
//  Created by Zhihao Li on 5/2/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
@testable import espresso

class SqueezeNetTests: XCTestCase {

  var network : Network!
  let filename = "/Users/Ben/Projects/espresso/models/squeezenet.espressomodel"

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
      kernelSize: 7,
      stride: 2
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv1",
      dependencies: ["conv1"]
      )))
    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool1",
      dependencies: ["relu_conv1"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire2_squeeze1x1",
      dependencies: ["pool1"],
      numOutput: 16,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire2_relu_squeeze1x1",
      dependencies: ["fire2_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire2_expand1x1",
      dependencies: ["fire2_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire2_relu_expand1x1",
      dependencies: ["fire2_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire2_expand3x3",
      dependencies: ["fire2_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire2_relu_expand3x3",
      dependencies: ["fire2_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire2_concat",
      dependencies: ["fire2_relu_expand1x1", "fire2_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire3_squeeze1x1",
      dependencies: ["fire2_concat"],
      numOutput: 16,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire3_relu_squeeze1x1",
      dependencies: ["fire3_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire3_expand1x1",
      dependencies: ["fire3_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire3_relu_expand1x1",
      dependencies: ["fire3_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire3_expand3x3",
      dependencies: ["fire3_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire3_relu_expand3x3",
      dependencies: ["fire3_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire3_concat",
      dependencies: ["fire3_relu_expand1x1", "fire3_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire4_squeeze1x1",
      dependencies: ["fire3_concat"],
      numOutput: 32,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire4_relu_squeeze1x1",
      dependencies: ["fire4_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire4_expand1x1",
      dependencies: ["fire4_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire4_relu_expand1x1",
      dependencies: ["fire4_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire4_expand3x3",
      dependencies: ["fire4_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire4_relu_expand3x3",
      dependencies: ["fire4_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire4_concat",
      dependencies: ["fire4_relu_expand1x1", "fire4_relu_expand3x3"]
      )))

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool4",
      dependencies: ["fire4_concat"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire5_squeeze1x1",
      dependencies: ["pool4"],
      numOutput: 32,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire5_relu_squeeze1x1",
      dependencies: ["fire5_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire5_expand1x1",
      dependencies: ["fire5_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire5_relu_expand1x1",
      dependencies: ["fire5_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire5_expand3x3",
      dependencies: ["fire5_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire5_relu_expand3x3",
      dependencies: ["fire5_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire5_concat",
      dependencies: ["fire5_relu_expand1x1", "fire5_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire6_squeeze1x1",
      dependencies: ["fire5_concat"],
      numOutput: 48,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire6_relu_squeeze1x1",
      dependencies: ["fire6_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire6_expand1x1",
      dependencies: ["fire6_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire6_relu_expand1x1",
      dependencies: ["fire6_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire6_expand3x3",
      dependencies: ["fire6_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire6_relu_expand3x3",
      dependencies: ["fire6_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire6_concat",
      dependencies: ["fire6_relu_expand1x1", "fire6_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire7_squeeze1x1",
      dependencies: ["fire6_concat"],
      numOutput: 48,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire7_relu_squeeze1x1",
      dependencies: ["fire7_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire7_expand1x1",
      dependencies: ["fire7_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire7_relu_expand1x1",
      dependencies: ["fire7_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire7_expand3x3",
      dependencies: ["fire7_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire7_relu_expand3x3",
      dependencies: ["fire7_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire7_concat",
      dependencies: ["fire7_relu_expand1x1", "fire7_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire8_squeeze1x1",
      dependencies: ["fire7_concat"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire8_relu_squeeze1x1",
      dependencies: ["fire8_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire8_expand1x1",
      dependencies: ["fire8_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire8_relu_expand1x1",
      dependencies: ["fire8_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire8_expand3x3",
      dependencies: ["fire8_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire8_relu_expand3x3",
      dependencies: ["fire8_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire8_concat",
      dependencies: ["fire8_relu_expand1x1", "fire8_relu_expand3x3"]
      )))

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool8",
      dependencies: ["fire8_concat"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire9_squeeze1x1",
      dependencies: ["pool8"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire9_relu_squeeze1x1",
      dependencies: ["fire9_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire9_expand1x1",
      dependencies: ["fire9_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire9_relu_expand1x1",
      dependencies: ["fire9_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire9_expand3x3",
      dependencies: ["fire9_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire9_relu_expand3x3",
      dependencies: ["fire9_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire9_concat",
      dependencies: ["fire9_relu_expand1x1", "fire9_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv10",
      dependencies: ["fire9_concat"],
      numOutput: 1000,
      kernelSize: 1,
      stride: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv10",
      dependencies: ["conv10"]
      )))
    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool10",
      dependencies: ["relu_conv10"],
      method: .AVG,
      globalPooling: true
      )))
    network.add(SoftmaxLayer(parameters: SoftmaxParameters(
      name: "prob",
      dependencies: ["pool10"]
      )))
  }

  override func tearDown() {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testPerformanceExample() {
    // This is an example of a performance test case.
    self.network.importFromFile(filename)
    self.measureBlock {
      self.network.forward()
    }
    print((network.layers.last as! ForwardLayerProtocol).output.storage)
  }
}
