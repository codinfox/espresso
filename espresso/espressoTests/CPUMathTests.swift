//
//  CPUMathTests.swift
//  espresso
//
//  Created by Zhihao Li on 5/8/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import XCTest

class CPUMathTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testExample() {
      let sparse = CompressedInfo(codebook: [1,2,3,4,6], spm: [UInt8(0), UInt8(1), UInt8(2), UInt8(1), UInt8(3), UInt8(4), UInt8(0), UInt8(1), UInt8(2)], ind: [0,1,2,3,4,5,6,7,8])
      let dense = [Float32](count: 9, repeatedValue: 1)
      let result = sparseDenseMatrixMultiplication(sparse, dense: dense, M: 3, N: 3, P: 3, groupOffset: 0)
      let expected : [Float32] = [6,6,6,12,12,12,6,6,6]
      XCTAssertEqual(result, expected)
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }

}
