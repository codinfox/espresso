//
//  Solver.swift
//  espresso
//
//  Created by Zhihao Li on 4/13/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief Base solver.
 */
protocol Solver {
    var name:String {get}
    var parameters:[String:Double] {get set}
    var net:Network{get set}
}