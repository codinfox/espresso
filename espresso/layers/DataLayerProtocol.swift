//
//  DataLayerProtocol.swift
//  espresso
//
//  Created by Zhihao Li on 4/16/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

protocol DataLayerProtocol : ForwardLayerProtocol {
  var batchSize : Int { set get }
}