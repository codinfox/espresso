//
//  espressoBlas.swift
//  espresso
//
//  Created by Jerry Zhang on 5/8/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

public func espressoSgemm(transA: Bool,
                          transB: Bool,
                          m: UInt32,
                          n: UInt32,
                          k: UInt32,
                          alpha: Float,
                          A: Tensor,
                          B: Tensor,
                          beta: Float,
                          C: Tensor,
                          metalDevice: MTLDevice,
                          metalDefaultLibrary: MTLLibrary,
                          metalCommandQueue: MTLCommandQueue) {

  let count = m * n
  let param = createSgemmParameter(MetalSgemmParameter(count: count, transA: transA, transB: transB, m: m, n: n, k: k, alpha: alpha, beta: beta), metalDevice: metalDevice)
  let commandBuffer = metalCommandQueue.commandBuffer()
  let (computeCommandEncoder, computePipelineState) = setupComputEncoder("espressoSgemm", commandBuffer: commandBuffer, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
  computeCommandEncoder.setBuffer(A.mtlStorage, offset: 0, atIndex: 0)
  computeCommandEncoder.setBuffer(B.mtlStorage, offset: 0, atIndex: 1)
  computeCommandEncoder.setBuffer(C.mtlStorage, offset: 0, atIndex: 2)
  computeCommandEncoder.setBuffer(param, offset: 0, atIndex: 3)
  submitComputeJob(computeCommandEncoder, computePipelineState: computePipelineState, count: Int(count))
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}


public func appleMM(m: UInt16,
                    n: UInt16,
                    k: UInt16,
                    A: Tensor,
                    B: Tensor,
                    C: Tensor,
                    metalDevice: MTLDevice,
                    metalDefaultLibrary: MTLLibrary,
                    metalCommandQueue: MTLCommandQueue) {
  let M = (m % 8 != 0) ? (m + 8) / 8 * 8 : m
  let K = (k % 8 != 0) ? (k + 8) / 8 * 8 : k
  let param = createAppleMMParameter(MetalMatrixDim(m: m, k: k, n: n, pbytes: M * UInt16(sizeof(Float)), qbytes: K * UInt16(sizeof(Float))), metalDevice: metalDevice)
  let commandBuffer = metalCommandQueue.commandBuffer()
  let (computeCommandEncoder, computePipelineState) = setupComputEncoder("MatrixMultiply", commandBuffer: commandBuffer, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
  computeCommandEncoder.setBuffer(A.mtlStorage, offset: 0, atIndex: 0)
  computeCommandEncoder.setBuffer(B.mtlStorage, offset: 0, atIndex: 1)
  computeCommandEncoder.setBuffer(C.mtlStorage, offset: 0, atIndex: 2)
  computeCommandEncoder.setBuffer(param, offset: 0, atIndex: 3)
  let width = (m % 8 != 0) ? (m + 8) / 8 : m / 8;
  let threadWidth: UInt16 = 4
  let threadGroupWidth = (width % threadWidth != 0)
    ? (width + threadWidth)/threadWidth : (width / threadWidth)

  let height = (k % 8 != 0) ? (k + 8) / 8 : k / 8
  let threadHeight: UInt16 = 8
  let threadGroupHeight = (height % threadHeight != 0)
    ? (height + threadHeight)/threadHeight
    : height / threadHeight;
  let threadsPerGroup = MTLSize(width: 4, height: 8, depth: 1)
  let numGroups = MTLSize(width: Int(threadGroupWidth), height: Int(threadGroupHeight), depth: 1)
  computeCommandEncoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

}