//
//  MetalUtilityFunctions.swift
//  espresso
//
//  Created by Jerry Zhang on 4/17/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal


/* Creating buffer utilities */
func createFloatArray(array: [Float], metalDevice: MTLDevice) -> MTLBuffer {
  let length = array.count * sizeof(Float)
  return metalDevice.newBufferWithBytes(array, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createReluParam(param: MetalReluParameter, metalDevice: MTLDevice) -> MTLBuffer {
  var param = param
  let length = sizeof(MetalReluParameter)
  return metalDevice.newBufferWithBytes(&param, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createConvolutionParameter(param: MetalConvolutionParameter, metalDevice: MTLDevice) -> MTLBuffer {
  var param = param
  let length = sizeof(MetalConvolutionParameter)
  return metalDevice.newBufferWithBytes(&param, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createFullyConnectedParameter(param: MetalFullyConnectedParameter, metalDevice: MTLDevice) -> MTLBuffer {
  var param = param
  let length = sizeof(MetalFullyConnectedParameter)
  return metalDevice.newBufferWithBytes(&param, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createSoftmaxParameter(param: MetalSoftmaxParameter, metalDevice: MTLDevice) -> MTLBuffer {
  var param = param
  let length = sizeof(MetalSoftmaxParameter)
  return metalDevice.newBufferWithBytes(&param, length: length, options: MTLResourceOptions.CPUCacheModeDefaultCache)
}

func createPoolingParameter(param: MetalPoolingParameter, metalDevice: MTLDevice) -> MTLBuffer {
  var param = param
  let length = sizeof(MetalPoolingParameter)
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

func setupComputEncoder(funcName: String, commandBuffer: MTLCommandBuffer, metalDefaultLibrary: MTLLibrary, metalDevice: MTLDevice) -> (MTLComputeCommandEncoder, MTLComputePipelineState){
  /* Setup the kernel function */
  let computeCommandEncoder = commandBuffer.computeCommandEncoder()
  let computePipelineState = createComputePipeline(funcName, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
  computeCommandEncoder.setComputePipelineState(computePipelineState)
  return (computeCommandEncoder, computePipelineState)
}

func submitComputeJob(computeCommandEncoder: MTLComputeCommandEncoder, computePipelineState: MTLComputePipelineState, count: Int) {
  let threadsPerGroup = MTLSize(width: min(count, computePipelineState.threadExecutionWidth), height: 1, depth: 1)
  let numGroups = MTLSize(width: count / computePipelineState.threadExecutionWidth + 1, height: 1, depth: 1)
  computeCommandEncoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
  computeCommandEncoder.endEncoding()
}

func submitCommonComputeJob(funcName: String, paramBuffer: MTLBuffer, metalDefaultLibrary: MTLLibrary, metalDevice: MTLDevice, inputData: Tensor, outputData: Tensor, commandBuffer: MTLCommandBuffer, threadCount: Int) {
  /* Setup the kernel function */
  let (computeCommandEncoder, computePipelineState) = setupComputEncoder(funcName, commandBuffer: commandBuffer, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)

  /* Copy data to gpu */
  computeCommandEncoder.setBuffer(inputData.mtlStorage, offset: 0, atIndex: 0)
  computeCommandEncoder.setBuffer(outputData.mtlStorage, offset: 0, atIndex: 1)
  computeCommandEncoder.setBuffer(paramBuffer, offset: 0, atIndex: 2)

  submitComputeJob(computeCommandEncoder, computePipelineState: computePipelineState, count: threadCount)
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
}