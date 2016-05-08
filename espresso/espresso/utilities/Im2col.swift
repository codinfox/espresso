//
//  Im2col.swift
//  espresso
//
//  Created by Jerry Zhang on 5/3/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation
import Metal

/** @brief image to column
 *  dimension
 *    rows: inputChannel * kernelSize * kernelSize
 *    cols: outputHeight * outputWidth
 *    transpose of the matrix in slide: http://15418.courses.cs.cmu.edu/spring2016/lecture/dnneval/slide_031
 *    The matrix on the right should be on the left, with dimension:
 *      numFilters * (numChannels * kernelSize * kernelSize)
 */
public func im2colCpu(input: [Tensor.DataType],
                      inputChannels: Int,
                      height: Int,
                      width: Int,
                      kernelSize: Int,
                      padSize: Int,
                      stride: Int,
                      inputOffset: Int = 0) -> [Tensor.DataType] {
    // change input/output to Tensor?
  let outputHeight = (height + 2 * padSize - kernelSize) / stride + 1
  let outputWidth = (width + 2 * padSize - kernelSize) / stride + 1

  let outputSize = outputHeight * outputWidth * inputChannels * kernelSize * kernelSize
  var output:[Float] = Array(count: outputSize, repeatedValue: 0)

  let channelSize = height * width
  var outputIndex = 0
  for curChan in 0..<inputChannels {
    for kernRow in 0..<kernelSize {
      for kernCol in 0..<kernelSize {
        var inputRow = kernRow - padSize
        // first row of output
        for _ in 0..<outputHeight { // outputRow
          var inputCol = kernCol - padSize
          for _ in 0..<outputWidth { // outputCol
            if inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width {
              output[outputIndex] = input[inputOffset + curChan * channelSize + inputRow * width + inputCol]
            } else {
              output[outputIndex] = 0
            }
            inputCol += stride
            outputIndex += 1
          }
          inputRow += stride
        }
      }
    }
  }
  return output
}


public func im2colGpu(input: Tensor,
                      output: Tensor,
                      inputChannels: Int,
                      height: Int,
                      width: Int,
                      kernelSize: Int,
                      padSize: Int,
                      stride: Int,
                      metalDevice: MTLDevice,
                      metalCommandQueue: MTLCommandQueue,
                      metalDefaultLibrary: MTLLibrary,
                      inputOffset: Int = 0) {
  let outputHeight = (height + 2 * padSize - kernelSize) / stride + 1
  let outputWidth = (width + 2 * padSize - kernelSize) / stride + 1
  let count = outputHeight * outputWidth * inputChannels * kernelSize * kernelSize
  let paramBuffer = createIm2colParameter(MetalIm2colParameter(count: Int32(count), inputOffset: Int32(inputOffset), kernelSize: Int32(kernelSize), padSize: Int32(padSize), stride: Int32(stride), inputChannels: Int32(inputChannels), inputHeight: Int32(height), inputWidth: Int32(width), outputHeight: Int32(outputHeight), outputWidth: Int32(outputWidth)), metalDevice: metalDevice)

  let commandBuffer = metalCommandQueue.commandBuffer()
  submitCommonComputeJob("im2colGpu", paramBuffer: paramBuffer, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice, inputData: input, outputData: output, commandBuffer: commandBuffer, threadCount: Int(count))
}
