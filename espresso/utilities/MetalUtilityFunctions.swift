//
//  MetalUtilityFunctions.swift
//  espresso
//
//  Created by Jerry Zhang on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal


func createFloatArray(array: [Float], metalDevice: MTLDevice) -> MTLBuffer {
  let length = array.count * sizeof(Float)
  return metalDevice.newBufferWithBytes(array, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createReluParam(param: MetalReluParameter, metalDevice: MTLDevice) -> MTLBuffer {
  var param = param
  let length = sizeof(MetalReluParameter)
  return metalDevice.newBufferWithBytes(&param, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}


func createComputePipeline(funcName: String, metalDefaultLibrary: MTLLibrary, metalDevice: MTLDevice) -> MTLComputePipelineState! {
  let function = metalDefaultLibrary.newFunctionWithName(funcName)
  var computePipelineState: MTLComputePipelineState?
  do {
    computePipelineState = try metalDevice.newComputePipelineStateWithFunction(function!)
  } catch {
    print ("Creating new compute pipeline state with function \(funcName) failed!")
  }
  return computePipelineState
}

func submitComputeJob(funcName: String, paramBuffer: MTLBuffer, metalDefaultLibrary: MTLLibrary, metalDevice: MTLDevice, inputData: Tensor, outputData: Tensor, commandBuffer: MTLCommandBuffer) -> MTLCommandBuffer {
  /* Setup the kernel function */
  let computeCommandEncoder = commandBuffer.computeCommandEncoder()
  let computePipelineState = createComputePipeline(funcName, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
  computeCommandEncoder.setComputePipelineState(computePipelineState)

  /* Copy data to gpu */
  computeCommandEncoder.setBuffer(inputData.mtlStorage, offset: 0, atIndex: 0)
  computeCommandEncoder.setBuffer(outputData.mtlStorage, offset: 0, atIndex: 1)
  computeCommandEncoder.setBuffer(paramBuffer, offset: 0, atIndex: 2)
  let threadsPerGroup = MTLSize(width: computePipelineState.threadExecutionWidth, height: 1, depth: 1)
  let count = inputData.storage.count / sizeof(Float)
  let numThreads = MTLSize(width: count / computePipelineState.threadExecutionWidth, height: 1, depth: 1)
  computeCommandEncoder.dispatchThreadgroups(numThreads, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
  commandBuffer.commit()
  return commandBuffer
}