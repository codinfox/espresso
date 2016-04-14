//
//  SgdSolver.swift
//  espresso
//
//  Created by Jerry Zhang on 4/14/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import Foundation

/** @brief The stochastic gradient descent solver.
 */
class SgdSolver:Solver {
    var name = "SGD Solver"
    var parameters: [String : Double]
    var net: Net = Net()
    init(parameters:[String:Double], net:Net) {
        self.parameters = parameters
        self.net = net
    }
}