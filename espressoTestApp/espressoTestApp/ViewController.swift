//
//  ViewController.swift
//  espressoTestApp
//
//  Created by Jerry Zhang on 5/6/16.
//  Copyright © 2016 Jerry Zhang. All rights reserved.
//

import UIKit
import espresso
import Metal

class ViewController: UIViewController {

  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view, typically from a nib.
    TensorTestCase().testTensor()
    ImageDataTestCase().testImageData()
    ReluTestCase().testRelu()
    ConvTestCase().testConv()
    PoolingTestCase().testPooling()
    SoftmaxTestCase().testSoftmax()
    //FullyConnectedTestCase().testFullyConnected()
    ConcatTestCase().testConcat()
    print("All layers tested.")
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }


}

class TensorTestCase {

  var tensor : Tensor?

  func testReshape() {
    let dim : [Int] = [3,4,4]
    tensor = Tensor(metalDevice: MTLCreateSystemDefaultDevice())
    // Reshape from an empty tensor
    tensor?.reshape(dim)

    assert((tensor?.dimensions)! == dim)
    assert((tensor?.numel)! == 3*4*4)
    //print((tensor?.indexAuxilary)! == [4*4, 4, 1])

    // Should not do anything
    tensor?.reshape(dim)

    assert((tensor?.dimensions)! == dim)
    assert((tensor?.numel)! == 3*4*4)
    //XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Shape back from [3,7,7]
    tensor?.reshape(dim)

    assert((tensor?.dimensions)! == dim)
    assert((tensor?.numel)! == 3*4*4)
    //XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Reshape to null tensor
    tensor?.reshape([])
    assert((tensor?.dimensions)! == [])
    assert((tensor?.numel)! == 0)
  }


  func testTensor() {
    testReshape()
    print("Tensor: All tests finished.")
  }
}

class ReluTestCase {
  let params = ReLUParameters(name: "Relu Layer Test",
                              dependencies: ["Conv Layer"],
                              negativeSlope: 2)
  var networkProp: NetworkProperties!
  let bottomDimensions = [[2,3,4,4]]
  var layer: ReluLayer!
  var metalDevice: MTLDevice!
  //var layer : ReluLayer = ReluLayer(parameters: params)

  func initialize() {
    layer = ReluLayer(parameters: params)
    // Initialize gpu
    metalDevice = MTLCreateSystemDefaultDevice()

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    networkProp = NetworkProperties(batchSize: 2, engine: .GPU)

    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    layer.reshapeByBottomDimensions(bottomDimensions)
  }

  func testLayerSetUp() {
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions[0])!, networkProp.batchSize)
    XCTAssertEqual((layer?.gradient.dimensions[0])!, networkProp.batchSize)
  }

  func testReshape() {
    let batchSize = 5
    let bottomNumOutput = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    layer.reshapeByBottomDimensions(bottomDimensions)
    print("Relu: testReshape [" + ((layer?.output.dimensions)! == bottomDimensions[0]).description + "]")
    //XCTAssertEqual((layer?.gradient.dimensions)!, bottomDimensions[0])
  }

  func testForwardGPU() {
    let bottom = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[0])
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

    bottom.putToMetal()
    layer.forwardGPU([bottom])

    layer.output.getFromMetal()
    let output = layer.output
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
    print("Relu: testForwardGPU [" + (output!.storage == expected).description + "]")
  }

  func testRelu() {
    initialize()
    testLayerSetUp()
    testReshape()
    initialize()
    testForwardGPU()
    print("Relu: All tests finished.")
  }
}

class ConvTestCase {
  var layer : ConvolutionLayer!
  let params = ConvolutionParameters(name: "Conv Layer Test",
                                     dependencies: ["Image Data Layer"],
                                     numOutput: 2,
                                     kernelSize: 2,
                                     stride: 2,
                                     padSize: 2,
                                     isBiasTerm: true)
  var metalDevice: MTLDevice!
  let networkProp = NetworkProperties(batchSize: 1, engine: .GPU)
  let bottomDimensions = [[1,3,4,4]]

  func initialize() {
    layer = ConvolutionLayer(parameters: params)
    // Initialize gpu
    metalDevice = MTLCreateSystemDefaultDevice()

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    layer.reshapeByBottomDimensions(bottomDimensions)
  }

  func testLayerSetUp() {
    let channels = params.numOutput
    let kernelSize = params.kernelSize
    let bottomChannels = bottomDimensions[0][1]
    let weightDim = [channels, bottomChannels, kernelSize, kernelSize]
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.weights.dimensions)!, weightDim)
    XCTAssertEqual((layer?.bias.dimensions)!, [channels])
    XCTAssertEqual((layer?.output.dimensions[0])!, networkProp.batchSize)
    //XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  }

  func testReshape() {
    let batchSize = 1
    let bottomNumOutput = 3
    //let chan = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    let outHeight = (height + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    let outWidth = (width + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, [batchSize, params.numOutput, outHeight, outWidth])
    XCTAssertEqual((layer?.gradient.dimensions)!, [batchSize, params.numOutput, outHeight, outWidth])
  }

  func testForwardGPU() {
    let bottom = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[0])
    bottom.storage = [ 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16]
    bottom.putToMetal()

    layer?.weights.storage = [1,1,
                              1,1,
                              1,1,
                              1,1,
                              0,0,
                              0,0,

                              2, 2,
                              2, 2,
                              -1,-1,
                              -1,-1,
                              1, 1,
                              1, 1]
    layer?.weights.putToMetal()

    layer?.bias.storage = [100,200]
    layer?.bias.putToMetal()

    layer?.forwardGPU([bottom])

    layer?.output.getFromMetal()
    let output = layer?.output
    XCTAssertEqual(output!.storage, [
      100,100,100,100,
      100,128,144,100,
      100,192,208,100,
      100,100,100,100,

      200,200,200,200,
      200,228,244,200,
      200,292,308,200,
      200,200,200,200
      ])

  }

  func testForwardGPU2() {
    let params2 = ConvolutionParameters(name: "Conv Layer Test",
                                        dependencies: ["Image Data Layer"],
                                        numOutput: 2,
                                        kernelSize: 3,
                                        stride: 2,
                                        padSize: 2,
                                        isBiasTerm: true)

    layer = ConvolutionLayer(parameters: params2)
    // Initialize gpu
    metalDevice = MTLCreateSystemDefaultDevice()

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    layer.reshapeByBottomDimensions(bottomDimensions)

    let bottom = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[0])
    bottom.storage = [ 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16]
    bottom.putToMetal()

    layer?.weights.storage = [1,1,1,
                              1,1,1,
                              1,1,1,

                              1,1,1,
                              0,0,0,
                              0,0,0,

                              0,0,0,
                              1,1,1,
                              0,0,0,

                              1,1,1,
                              1,1,1,
                              1,1,1,

                              1,1,1,
                              0,0,0,
                              0,0,0,

                              0,0,0,
                              1,1,1,
                              0,0,0]


    layer?.weights.putToMetal()

    layer?.bias.storage = [100,200]
    layer?.bias.putToMetal()

    layer?.forwardGPU([bottom])

    layer?.output.getFromMetal()
    let output = layer?.output
    XCTAssertEqual(output!.storage, [
      101, 106, 107,
      121, 178, 167,
      144, 244, 208,

      201, 206, 207,
      221, 278, 267,
      244, 344, 308
      ])
  }

  func testConv() {
    initialize()
    testLayerSetUp()
    testReshape()
    initialize()
    testForwardGPU()
    testForwardGPU2()
    print("Conv: All tests finished.")
  }

}

class PoolingTestCase {

  var layer : PoolingLayer!
  let params = PoolingParameters(name: "Pooling Layer Test",
                                 dependencies: ["Conv Layer"], /* whatever */
    kernelSize: 3,
    stride: 2,
    padSize: 2,
    method: .MAX)
  var metalDevice: MTLDevice!
  let networkProp = NetworkProperties(batchSize: 1, engine: .GPU)
  let bottomDimensions = [[1,3,4,4]]

  func initialize() {
    layer = PoolingLayer(parameters: params)
    //    layer = PoolingLayer(parameters: PoolingParameters(
    //      name: "Pooling Layer Test",
    //      dependencies: ["Conv Layer"], /* whatever */
    //      method: .AVG,
    //      globalPooling: true
    //      ))

    // Initialize gpu
    metalDevice = MTLCreateSystemDefaultDevice()

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    layer.reshapeByBottomDimensions(bottomDimensions)
  }

  func testLayerSetUp() {
    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions[0])!, networkProp.batchSize)
    //XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  }

  func testReshape() {
    let batchSize = 3
    let bottomNumOutput = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    let outHeight = (height + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    let outWidth = (width + params.padSize * 2 - params.kernelSize + params.stride) / params.stride
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, [batchSize, bottomNumOutput, outHeight, outWidth])
    //XCTAssertEqual((layer?.gradient.dimensions)!, [batchSize, bottomNumOutput, outHeight, outWidth])
  }

  func testForwardGPU() {
    let batchSize = 1
    let chan = 3
    let height = 4
    let width = 4
    let bottomDimensions = [[batchSize, chan, height, width]]
    let bottom = Tensor(metalDevice: metalDevice, dimensions: [batchSize, chan, height, width])
    bottom.storage = [ 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16]
    bottom.putToMetal()

    layer?.forwardGPU([bottom])
    layer.output.getFromMetal()
    let output = layer?.output
    let expected: [Float] = [
      1,3,4,
      9,11,12,
      13,15,16,

      1,3,4,
      9,11,12,
      13,15,16,

      1,3,4,
      9,11,12,
      13,15,16,
      ]
    XCTAssertEqual(output!.storage, expected)
  }

  func testGlobalPooling() {
    //    layer = PoolingLayer(parameters: PoolingParameters(
    //      name: "Pooling Layer Test",
    //      dependencies: ["Conv Layer"], /* whatever */
    //      method: .AVG,
    //      globalPooling: true
    //      ))
    // Queue to handle an ordered list of command buffers
    //    let metalCommandQueue = metalDevice!.newCommandQueue()
    //
    //    // Access to Metal functions that are stored in .metal file
    //    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()
    //
    //    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    //    layer.output.reshape([])
    //    layer.reshapeByBottomDimensions(bottomDimensions)


    let bottom = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[0])
    bottom.storage = [ 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16,

                       1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,16]
    bottom.putToMetal()

    layer?.forwardGPU([bottom])
    layer.output.getFromMetal()

    let output = layer?.output
    let expected: [Float] = [ 8.5, 8.5, 8.5 ]
    XCTAssertEqual(output!.storage, expected)
  }

  func testPooling() {
    /* initialize()
     testLayerSetUp() */
    // testReshape()
    initialize()
    testForwardGPU()
    //initialize()
    //testGlobalPooling()
    print("Pooling: All tests finished.")
  }
}

class FullyConnectedTestCase {
  //  var layer : FullyConnectedLayer!
  //  let params = FullyConnectedParameters(name: "FollyConnected Layer Test",
  //                                        dependencies: ["Conv Layer"],
  //                                        numOutput: 2,
  //                                        isBiasTerm: false,
  //                                        biasLRMultiplier: 0.5,
  //                                        weightLRMultiplier: 0.5,
  //                                        weightFiller: gaussianWeightFiller(mean: 0, std: 1),
  //                                        biasFiller: gaussianWeightFiller(mean: 0, std: 1))
  //  let network = NetworkProperties(batchSize: 2, engine: .CPU)
  //  let bottomDimensions = [[2,1,3,4]]
  //
  //  override func setUp() {
  //    super.setUp()
  //    // Put setup code here. This method is called before the invocation of each test method in the class.
  //    layer = FullyConnectedLayer(parameters: params)
  //  }
  //
  //  override func tearDown() {
  //    layer = nil
  //    // Put teardown code here. This method is called after the invocation of each test method in the class.
  //    super.tearDown()
  //  }
  //
  //  func testLayerSetUp() {
  //    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
  //    XCTAssertEqual(layer?.output.dimensions[0], network.batchSize)
  //    //XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  //  }
  //
  //  func testReshape() {
  //    let batchSize = 5
  //    let bottomNumOutput = 3
  //    let height = 12
  //    let width = 20
  //    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
  //    let outputDimension = [params.numOutput]
  //    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
  //    layer?.reshapeByBottomDimensions(bottomDimensions)
  //    XCTAssertEqual((layer?.output.dimensions)!, [batchSize, outputDimension[0]])
  //    //XCTAssertEqual((layer?.gradient.dimensions)!, outputDimensions)
  //  }
  //
  //  func testForwardGPU() {
  //    let batchSize = 2
  //    let chan = 3
  //    let height = 4
  //    let width = 4
  //    let bottomDimensions = [[batchSize, chan, height, width]]
  //    let bottom = Tensor(dimensions: [batchSize, chan, height, width])
  //    let network = NetworkProperties(batchSize: 2, engine: .CPU)
  //    layer?.layerSetUp(engine: network.engine, bottomDimensions: bottomDimensions)
  //
  //    bottom.storage =
  //      [ 1, 2, 3, 4, /* batch 0, channel 0 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4, /* batch 0, channel 1 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4, /* batch 0, channel 2 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4,  /* batch 1, channel 0 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4,  /* batch 1, channel 1 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4, /* batch 1, channel 2 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16]
  //
  //    layer?.weights.storage =
  //      [ 1, 2, 3, 4, /* output 0, channel 0 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4, /* output 0, channel 1 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, 2, 3, 4, /* output 0, channel 2 */
  //        5, 6, 7, 8,
  //        9,10,11,12,
  //        13,14,15,16,
  //
  //        1, -1, 1, -1, /* output 1, channel 0 */
  //        1, -1, 1, -1,
  //        1, -1, 1, -1,
  //        1, -1, 1, -1,
  //
  //        1, -1, 1, -1, /* output 1, channel 1 */
  //        1, -1, 1, -1,
  //        1, -1, 1, -1,
  //        1, -1, 1, -1,
  //
  //        1, -1, 1, -1, /* output 1, channel 2 */
  //        1, -1, 1, -1,
  //        1, -1, 1, -1,
  //        1, -1, 1, -1]
  //
  //    layer?.forwardCPU([bottom])
  //
  //    let output = layer?.output
  //    let expected:[Float] = [4488.0, -24.0]
  //    XCTAssertEqual(output!.storage, expected)
  //  }

}


class SoftmaxTestCase {
  var layer : SoftmaxLayer!
  let params = SoftmaxParameters(name: "Softmax Layer Test",
                                 dependencies: ["Conv Layer"], /* whatever */
    axis: 1)
  var metalDevice: MTLDevice!
  var networkProp = NetworkProperties(batchSize: 1, engine: .GPU)
  let bottomDimensions = [[1,3,4,4]]

  func initialize() {
    layer = SoftmaxLayer(parameters: params)
    // Initialize gpu
    metalDevice = MTLCreateSystemDefaultDevice()

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    networkProp = NetworkProperties(batchSize: 2, engine: .GPU)

    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    layer.reshapeByBottomDimensions(bottomDimensions)

  }

  func testLayerSetUp() {
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions[0])!, networkProp.batchSize)
    //XCTAssertEqual(layer?.gradient.dimensions[0], network.batchSize)
  }

  func testReshape() {
    let batchSize = 3
    let channels = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, channels, height, width]]
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    layer?.reshapeByBottomDimensions(bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions)!, bottomDimensions[0])
    //XCTAssertEqual((layer?.gradient.dimensions)!, outputDimension)
  }

  func testForwardGPU() {
    let bottom = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[0])
    bottom.storage = [ 1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 1,

                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,

                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0,
                       1, 0, 1, 0]
    bottom.putToMetal()
    layer?.forwardGPU([bottom])

    layer.output.getFromMetal()
    let output = layer?.output
    let Z:Float = exp(0) + exp(-1) + exp(-1)
    let expected: [Float] = [
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/Z,

      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, exp(-1)/Z,

      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, 1/3,
      1/3, 1/3, 1/3, exp(-1)/Z
    ]

    let error = (Array(0..<48).map{abs(expected[$0] - output!.storage[$0])}).maxElement()
    XCTAssertTrue(error < 0.00001, "Error greater than 0.00001")
  }

  func testSoftmax() {
    initialize()
    testForwardGPU()
  }
}

class ConcatTestCase {
  let params = ConcatParameters(name: "Concat Layer Test",
                                dependencies: ["Conv Layer"],
                                axis: 1)
  var networkProp: NetworkProperties!
  let bottomDimensions = [[1,2,3,3], [1,3,3,3]]
  var layer: ConcatLayer!
  var metalDevice: MTLDevice!

  func initialize() {
    layer = ConcatLayer(parameters: params)
    // Initialize gpu
    metalDevice = MTLCreateSystemDefaultDevice()

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    networkProp = NetworkProperties(batchSize: 2, engine: .GPU)

    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions, metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    layer.reshapeByBottomDimensions(bottomDimensions)
  }

  func testLayerSetUp() {
    layer?.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    XCTAssertEqual((layer?.output.dimensions[0])!, networkProp.batchSize)
    XCTAssertEqual((layer?.gradient.dimensions[0])!, networkProp.batchSize)
  }

  func testReshape() {
    let batchSize = 5
    let bottomNumOutput = 3
    let height = 12
    let width = 20
    let bottomDimensions = [[batchSize, bottomNumOutput, height, width]]
    layer.layerSetUp(engine: networkProp.engine, bottomDimensions: bottomDimensions)
    layer.reshapeByBottomDimensions(bottomDimensions)
    print("Relu: testReshape [" + ((layer?.output.dimensions)! == bottomDimensions[0]).description + "]")
    //XCTAssertEqual((layer?.gradient.dimensions)!, bottomDimensions[0])
  }

  func testForwardGPU() {
    let bottom0 = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[0])
    bottom0.storage = [
      1,2,3,
      4,5,6,
      7,8,9,

      1,2,3,
      4,5,6,
      7,8,9]

    bottom0.putToMetal()

    let bottom1 = Tensor(metalDevice: metalDevice, dimensions: bottomDimensions[1])
    bottom1.storage = [
      1,2,3,
      4,5,6,
      7,8,9,

      1,2,3,
      4,5,6,
      7,8,9,

      10,11,12,
      13,14,15,
      16,17,18]
    bottom1.putToMetal()

    layer.forwardGPU([bottom0, bottom1])

    layer.output.getFromMetal()
    let output = layer.output
    let expected:[Float] = [
      1,2,3,
      4,5,6,
      7,8,9,

      1,2,3,
      4,5,6,
      7,8,9,

      1,2,3,
      4,5,6,
      7,8,9,

      1,2,3,
      4,5,6,
      7,8,9,

      10,11,12,
      13,14,15,
      16,17,18]

    print("Concat: testForwardGPU [" + (output!.storage == expected).description + "]")
  }

  func testConcat() {
    //initialize()
    //testLayerSetUp()
    //testReshape()
    initialize()
    testForwardGPU()
    print("Concat: All tests finished.")
  }
}

class ImageDataTestCase {

  let metalDevice = MTLCreateSystemDefaultDevice()
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
    imgDataLayer.layerSetUp(engine: .GPU, bottomDimensions: [dimension])

    let expected = Tensor(dimensions: dimension)
    expected.storage = [1,2,3,4,
                        1,2,3,4,
                        1,2,3,4,
                        1,2,3,4,
                        1,2,3,4,
                        1,2,3,4]

    // Queue to handle an ordered list of command buffers
    let metalCommandQueue = metalDevice!.newCommandQueue()

    // Access to Metal functions that are stored in .metal file
    let metalDefaultLibrary = metalDevice!.newDefaultLibrary()

    imgDataLayer.layerSetUp(engine: .GPU, bottomDimensions: [dimension], metalDevice: metalDevice, metalDefaultLibrary: metalDefaultLibrary, metalCommandQueue: metalCommandQueue)
    imgDataLayer.reshapeByBottomDimensions([dimension])

    let output = imgDataLayer.output
    let length = output.count() * sizeof(Float)

    // get content from metal to swift
    let mtlContent = NSData(bytesNoCopy: output.mtlStorage.contents(),
                            length: length, freeWhenDone: false)
    var initialContentArray = [Float](count: output.count(), repeatedValue: 0)
    mtlContent.getBytes(&initialContentArray, length:length)
    
    let allZeros = [Float](count: expected.count(), repeatedValue: 0)
    XCTAssertEqual(initialContentArray, allZeros)
    
    imgDataLayer.forwardGPU([])
    let finalOutput = NSData(bytesNoCopy: output.mtlStorage.contents(),
                             length: length, freeWhenDone: false)
    var finalContentArray = [Float](count: output.count(), repeatedValue: 0)
    finalOutput.getBytes(&finalContentArray, length:length)
    
    
    XCTAssertEqual(finalContentArray, expected.storage)
  }
  
  func testImageData() {
    //initialize()
    //testLayerSetUp()
    //testReshape()
    //initialize()
    //testForwardGPU()
    testExample()
    print("Image Data: All tests finished.")
  }
}

func XCTAssertEqual(a: [Float], _ b: [Float]) {
  assert(a == b)
}

func XCTAssertEqual(a: Int, _ b: Int) {
  assert(a == b)
}

func XCTAssertEqual(a: [Int], _ b: [Int]) {
  assert(a == b)
}

func XCTAssertTrue(b: Bool, _ msg: String) {
  if !b {
    print(msg)
  }
}

