//
//  MathFunctions.swift
//  espresso
//
//  Created by Jerry Zhang on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Accelerate

public func sigmoid_cpu(x: Float) -> Float {
    return 1.0 / (1.0 + exp(x))
}

var concurrentQueue : dispatch_queue_t! = nil
var onceToken : dispatch_once_t = 0

/**
 Sparse: M * P
 Dense: P * N
 Output: M * N
 */
public func sparseDenseMatrixMultiplication(sparse: CompressedInfo, dense: [Float32], M: Int, N: Int, P: Int) -> [Float32] {
  dispatch_once(&onceToken) { 
    concurrentQueue = dispatch_queue_create("edu.cmu.espresso.blasqueue", DISPATCH_QUEUE_CONCURRENT)
  }

  var group = dispatch_group_create()
  var outputs : [[Float32]] = [[],[]]
  for i in 0 ..< 2 {
    dispatch_group_enter(group)
    dispatch_group_async(group, concurrentQueue, { 
      outputs[i] = sparseDenseMatrixMultiplicationHelper(sparse, dense: dense, M: M, N: N, P: P, startIndex: i)
      dispatch_group_leave(group)
    })
  }
  dispatch_group_wait(group, DISPATCH_TIME_FOREVER)
  vDSP_vadd(&outputs[0], 1, &outputs[1], 1, &outputs[0], 1, vDSP_Length(N*M))
  return outputs[0]
}

func sparseDenseMatrixMultiplicationHelper(sparse: CompressedInfo, dense: [Float32], M: Int, N: Int, P: Int, startIndex: Int) -> [Float32] {
  let ind = sparse.ind
  let spm = sparse.spm
  let codebook = sparse.codebook
  var dense = dense

  var output = [Float32](count: M * N, repeatedValue: 0)
  var outputPointer = 0

  var rowResult = [Float32](count: N, repeatedValue: 0)
  var prevRow = 0

  let firstRow = ind[0] / Int32(P)
  outputPointer = N * Int(firstRow)
  prevRow = Int(firstRow)

  var _ZERO = Float32(0)

  for idx in startIndex.stride(to: ind.count, by: 2) {
    let idx = Int(idx)
    let row = Int(ind[idx]) / P
    let col = Int(ind[idx]) % P
    var val = codebook[Int(spm[idx])]

    if row != prevRow {
      vDSP_vadd(&rowResult, 1, (&output + outputPointer), 1, (&output + outputPointer), 1, vDSP_Length(N))
      outputPointer += N
      vDSP_vfill(&_ZERO, &rowResult, 1, vDSP_Length(N))
    }

    prevRow = row

    let denseStart = col * N

    vDSP_vsma((&dense + denseStart), 1, &val, &rowResult, 1, &rowResult, 1, vDSP_Length(N))
  }
  vDSP_vadd(&rowResult, 1, (&output + outputPointer), 1, (&output + outputPointer), 1, vDSP_Length(N))
  return output
}