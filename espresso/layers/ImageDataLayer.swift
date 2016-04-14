//
//  ImageDataLayer.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The image data input layer.
 */
public class ImageDataLayer : DataLayer {
    let name: String = "ImageDataLayer"
    var data: Tensor<Int>
    init(imgFileName:String) {
        self.data = Tensor<Int>(dimensions:[])
    }
}