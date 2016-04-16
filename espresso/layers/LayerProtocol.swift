//
//  Layer.swift
//  espresso
//
//  Created by Zhihao Li on 4/13/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The base layer.
 */
public protocol LayerProtocol {
  var name: String { get set }
//  var parameters : LayerParameterProtocol { get set }
//  init(name: String, parameters : LayerParameterProtocol)
}

protocol LayerParameterProtocol {
  
}