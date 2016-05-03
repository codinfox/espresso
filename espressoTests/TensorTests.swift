//
//  TensorTests.swift
//  espresso
//
//  Created by Zhihao Li on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest
import Metal

@testable import espresso

class TensorTestCPU: XCTestCase {

  var tensor : Tensor? = nil

  override func setUp() {
    super.setUp()
    // Put setup code here. This method is called before the invocation of each test method in the class.
  }

  override func tearDown() {
    tensor = nil
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testReshape() {
    tensor = Tensor()
    let dim : [Int] = [3,4,4]

    // Reshape from an empty tensor
    tensor?.reshape(dim)

    XCTAssertEqual((tensor?.dimensions)!, dim)
    XCTAssertEqual((tensor?.numel)!, 3*4*4)
    XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Should not do anything
    tensor?.reshape(dim)

    XCTAssertEqual((tensor?.dimensions)!, dim)
    XCTAssertEqual((tensor?.numel)!, 3*4*4)
    XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Reshape when initialization
    tensor = Tensor(dimensions: [3,7,7])

    XCTAssertEqual((tensor?.dimensions)!, [3,7,7])
    XCTAssertEqual((tensor?.numel)!, 3*7*7)
    XCTAssertEqual((tensor?.indexAuxilary)!, [7*7, 7, 1])

    // Shape back from [3,7,7]
    tensor?.reshape(dim)

    XCTAssertEqual((tensor?.dimensions)!, dim)
    XCTAssertEqual((tensor?.numel)!, 3*4*4)
    XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Reshape to null tensor
    tensor?.reshape([])
    XCTAssertEqual((tensor?.dimensions)!, [])
    XCTAssertEqual((tensor?.numel)!, 0)
  }

  func testSubscript() {
    tensor = Tensor(dimensions: [3,4,4])
    tensor![1,1,1] = 2
    XCTAssertEqual(tensor![1,1,1], 2)
    XCTAssertEqual(tensor?.storage[21], 2)
  }

  func testReset() {
    tensor = Tensor(dimensions: [2,2,2])
    tensor![1,1,1] = 2
    XCTAssertEqual(tensor![1,1,1], 2)
    tensor?.reset(0)
    XCTAssertEqual(tensor!.storage, Array(count: 8, repeatedValue: 0))
  }

  func testCount() {
    tensor = Tensor()
    XCTAssertEqual(tensor?.count(), 1)
    XCTAssertEqual(tensor?.count(fromDimension: 1,toDimension: 2), 1)
    tensor = Tensor(dimensions: [2,2,3])
    XCTAssertEqual(tensor?.count(), 12)
    XCTAssertEqual(tensor?.count(fromDimension: 1), 6)
    XCTAssertEqual(tensor?.count(toDimension: 1), 4)
    XCTAssertEqual(tensor?.count(fromDimension: 1, toDimension: 1), 2)
  }
}

class TensorTestGPU: XCTestCase {
  var tensor : Tensor?

  override func setUp() {
    super.setUp()
    let metalDevice = MTLCreateSystemDefaultDevice()
    tensor = Tensor(metalDevice: metalDevice)
  }

  override func tearDown() {
    tensor = nil
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    super.tearDown()
  }

  func testReshape() {
    let dim : [Int] = [3,4,4]
    tensor = Tensor(metalDevice: MTLCreateSystemDefaultDevice())
    // Reshape from an empty tensor
    tensor?.reshape(dim, engine: .GPU)

    XCTAssertEqual((tensor?.dimensions)!, dim)
    XCTAssertEqual((tensor?.numel)!, 3*4*4)
    XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Should not do anything
    tensor?.reshape(dim, engine: .GPU)

    XCTAssertEqual((tensor?.dimensions)!, dim)
    XCTAssertEqual((tensor?.numel)!, 3*4*4)
    XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Shape back from [3,7,7]
    tensor?.reshape(dim, engine: .GPU)

    XCTAssertEqual((tensor?.dimensions)!, dim)
    XCTAssertEqual((tensor?.numel)!, 3*4*4)
    XCTAssertEqual((tensor?.indexAuxilary)!, [4*4, 4, 1])

    // Reshape to null tensor
    tensor?.reshape([], engine: .GPU)
    XCTAssertEqual((tensor?.dimensions)!, [])
    XCTAssertEqual((tensor?.numel)!, 0)
  }

}

