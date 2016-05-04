//
//  Im2col.swift
//  espresso
//
//  Created by Jerry Zhang on 5/3/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief image to column
 *  dimension
 *    height: channel * kernelSize * kernelSize
 *    width: paddedHeight * paddedWidth
 *    transpose of the matrix in slide: http://15418.courses.cs.cmu.edu/spring2016/lecture/dnneval/slide_031
 *    The matrix on the right should be on the lefe, with dimension: 
 *      numFilters * (numChannels * kernelSize * kernelSize)
 */
func im2colCpu(input: [Tensor.DataType], inputChannels: Int, height: Int, width: Int,
               kernelSize: Int, padSize: Int, stride: Int) -> [Tensor.DataType] {
  let outputSize = height * width * inputChannels * kernelSize * kernelSize
  var output:[Float] = Array(count: outputSize, repeatedValue: 0)
  let inputHeight = height + 2 * padSize
  let inputWidth = width + 2 * padSize
  let outputHeight = (height + 2 * padSize - kernelSize) / stride + 1
  let outputWidth = (width + 2 * padSize - kernelSize) / stride + 1
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
            if inputRow >= padSize && inputRow < inputHeight - padSize && inputCol >= padSize && inputCol < inputWidth - padSize {
              output[outputIndex] = input[curChan * channelSize + inputRow * width + inputCol]
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