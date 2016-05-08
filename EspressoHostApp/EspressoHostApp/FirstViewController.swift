//
//  FirstViewController.swift
//  EspressoHostApp
//
//  Created by Zhihao Li on 5/5/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import UIKit
import espresso

class FirstViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  var globalImage : UIImage! = nil

  func readUIImageToTensor() -> Tensor {
    let inputCGImage = globalImage!.CGImage//UIImage(contentsOfFile: NSBundle.mainBundle().pathForResource("example", ofType: "jpg")!)!.CGImage
    //globalImage.CGImage //UIImage(contentsOfFile: "/Users/Ben/Downloads/ING-bell-pepper_sql.jpg")!.CGImage
    let width = 227 // CGImageGetWidth(inputCGImage)
    let height = 227 // CGImageGetHeight(inputCGImage)

    let tensor = Tensor(dimensions: [1,3,height,width]) // FIXME: demo

    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let bitsPerComponent = 8

    let pixels = UnsafeMutablePointer<UInt32>(calloc(height * width, sizeof(UInt32)))
    let colorSpace = CGColorSpaceCreateDeviceRGB()

    let context = CGBitmapContextCreate(pixels, width, height, bitsPerComponent, bytesPerRow, colorSpace, CGImageAlphaInfo.PremultipliedFirst.rawValue)

    // will automatically resize image to [width, height]
    CGContextDrawImage(context, CGRectMake(0, 0, CGFloat(width), CGFloat(height)), inputCGImage)

    let dataPointer = UnsafePointer<UInt8>(pixels)

    for j in 0 ..< height {
      for i in 0 ..< width {
        let offset = 4*((Int(width) * Int(j)) + Int(i))
        //        let alphaValue = dataType[offset] as UInt8
        tensor[0,0,j,i] = Tensor.DataType(dataPointer[offset+3]) - 104 // blue (- mean)
        tensor[0,1,j,i] = Tensor.DataType(dataPointer[offset+2]) - 117 // green (- mean)
        tensor[0,2,j,i] = Tensor.DataType(dataPointer[offset+1]) - 123 // red (- mean)
      }
    }

    free(pixels)
    return tensor
  }

  var network : Network! = nil

  @IBOutlet weak var imageView: UIImageView!


  @IBOutlet weak var label1: UILabel!
  @IBOutlet weak var label2: UILabel!
  @IBOutlet weak var label3: UILabel!
  @IBOutlet weak var label4: UILabel!
  @IBOutlet weak var label5: UILabel!

  @IBAction func recognize(sender: AnyObject) {
    let displayLabel = [label1, label2, label3, label4, label5]

    let outMtl = network.forward()
    if network.parameters.engine == .GPU {
      outMtl.getFromMetal()
    }
    let out = outMtl.storage
    var top5 = [Float]()
    var top5min : Int = 0

    for outidx in out.indices {
      let elem = out[outidx]
      if top5.count < 5 {
        top5.append(elem)
        if top5[top5min] > elem {
          top5min = top5.count - 1
        }
      } else {
        if elem > top5[top5min] {
          top5[top5min] = elem
          for tidx in top5.indices {
            if top5[tidx] < top5[top5min] {
              top5min = tidx
            }
          }
        }
      }
    }

    top5 = top5.sort().reverse()

    for idx in top5.indices {
      let index = out.indexOf(top5[idx])!
      displayLabel[idx].text = "\(top5[idx]) \(globalLabels[index])"
    }

  }
  @IBAction func takePicture(sender: AnyObject) {
    let picker = UIImagePickerController()
    picker.delegate = self
    picker.allowsEditing = true
    picker.sourceType = UIImagePickerControllerSourceType.Camera

    self.presentViewController(picker, animated: true, completion: nil)
  }

  func imagePickerController(picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : AnyObject]) {
    let chosenImage = info[UIImagePickerControllerEditedImage]
    self.imageView.image = chosenImage as? UIImage
    globalImage = chosenImage as? UIImage
    picker.dismissViewControllerAnimated(true, completion: nil)
  }

  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view, typically from a nib.
    let engine: NetworkProperties.NetworkEngine = .CPU
    network = Network(parameters: NetworkProperties(batchSize: 1, engine: engine))
    network.add(ImageDataLayer(parameters: ImageDataParameters(
      name: "data",
      imgNames: Array<String>(count: 100, repeatedValue: ""),
      dimensions: [1,3,227,227],
      dependencies: [],
      readImage: { _ in (self.readUIImageToTensor().storage, [0])}
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv1",
      dependencies: ["data"],
      numOutput: 96,
      kernelSize: 7,
      stride: 2
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv1",
      dependencies: ["conv1"]
      )))
    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool1",
      dependencies: ["relu_conv1"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire2_squeeze1x1",
      dependencies: ["pool1"],
      numOutput: 16,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire2_relu_squeeze1x1",
      dependencies: ["fire2_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire2_expand1x1",
      dependencies: ["fire2_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire2_relu_expand1x1",
      dependencies: ["fire2_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire2_expand3x3",
      dependencies: ["fire2_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire2_relu_expand3x3",
      dependencies: ["fire2_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire2_concat",
      dependencies: ["fire2_relu_expand1x1", "fire2_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire3_squeeze1x1",
      dependencies: ["fire2_concat"],
      numOutput: 16,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire3_relu_squeeze1x1",
      dependencies: ["fire3_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire3_expand1x1",
      dependencies: ["fire3_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire3_relu_expand1x1",
      dependencies: ["fire3_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire3_expand3x3",
      dependencies: ["fire3_relu_squeeze1x1"],
      numOutput: 64,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire3_relu_expand3x3",
      dependencies: ["fire3_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire3_concat",
      dependencies: ["fire3_relu_expand1x1", "fire3_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire4_squeeze1x1",
      dependencies: ["fire3_concat"],
      numOutput: 32,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire4_relu_squeeze1x1",
      dependencies: ["fire4_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire4_expand1x1",
      dependencies: ["fire4_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire4_relu_expand1x1",
      dependencies: ["fire4_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire4_expand3x3",
      dependencies: ["fire4_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire4_relu_expand3x3",
      dependencies: ["fire4_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire4_concat",
      dependencies: ["fire4_relu_expand1x1", "fire4_relu_expand3x3"]
      )))

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool4",
      dependencies: ["fire4_concat"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire5_squeeze1x1",
      dependencies: ["pool4"],
      numOutput: 32,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire5_relu_squeeze1x1",
      dependencies: ["fire5_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire5_expand1x1",
      dependencies: ["fire5_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire5_relu_expand1x1",
      dependencies: ["fire5_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire5_expand3x3",
      dependencies: ["fire5_relu_squeeze1x1"],
      numOutput: 128,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire5_relu_expand3x3",
      dependencies: ["fire5_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire5_concat",
      dependencies: ["fire5_relu_expand1x1", "fire5_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire6_squeeze1x1",
      dependencies: ["fire5_concat"],
      numOutput: 48,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire6_relu_squeeze1x1",
      dependencies: ["fire6_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire6_expand1x1",
      dependencies: ["fire6_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire6_relu_expand1x1",
      dependencies: ["fire6_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire6_expand3x3",
      dependencies: ["fire6_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire6_relu_expand3x3",
      dependencies: ["fire6_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire6_concat",
      dependencies: ["fire6_relu_expand1x1", "fire6_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire7_squeeze1x1",
      dependencies: ["fire6_concat"],
      numOutput: 48,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire7_relu_squeeze1x1",
      dependencies: ["fire7_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire7_expand1x1",
      dependencies: ["fire7_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire7_relu_expand1x1",
      dependencies: ["fire7_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire7_expand3x3",
      dependencies: ["fire7_relu_squeeze1x1"],
      numOutput: 192,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire7_relu_expand3x3",
      dependencies: ["fire7_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire7_concat",
      dependencies: ["fire7_relu_expand1x1", "fire7_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire8_squeeze1x1",
      dependencies: ["fire7_concat"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire8_relu_squeeze1x1",
      dependencies: ["fire8_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire8_expand1x1",
      dependencies: ["fire8_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire8_relu_expand1x1",
      dependencies: ["fire8_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire8_expand3x3",
      dependencies: ["fire8_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire8_relu_expand3x3",
      dependencies: ["fire8_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire8_concat",
      dependencies: ["fire8_relu_expand1x1", "fire8_relu_expand3x3"]
      )))

    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool8",
      dependencies: ["fire8_concat"],
      kernelSize: 3,
      stride: 2,
      method: .MAX
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire9_squeeze1x1",
      dependencies: ["pool8"],
      numOutput: 64,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire9_relu_squeeze1x1",
      dependencies: ["fire9_squeeze1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire9_expand1x1",
      dependencies: ["fire9_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire9_relu_expand1x1",
      dependencies: ["fire9_expand1x1"]
      )))
    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "fire9_expand3x3",
      dependencies: ["fire9_relu_squeeze1x1"],
      numOutput: 256,
      kernelSize: 3,
      padSize: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "fire9_relu_expand3x3",
      dependencies: ["fire9_expand3x3"]
      )))
    network.add(ConcatLayer(parameters: ConcatParameters(
      name: "fire9_concat",
      dependencies: ["fire9_relu_expand1x1", "fire9_relu_expand3x3"]
      )))

    network.add(ConvolutionLayer(parameters: ConvolutionParameters(
      name: "conv10",
      dependencies: ["fire9_concat"],
      numOutput: 1000,
      kernelSize: 1,
      stride: 1
      )))
    network.add(ReluLayer(parameters: ReLUParameters(
      name: "relu_conv10",
      dependencies: ["conv10"]
      )))
    network.add(PoolingLayer(parameters: PoolingParameters(
      name: "pool10",
      dependencies: ["relu_conv10"],
      method: .AVG,
      globalPooling: true
      )))
    network.add(SoftmaxLayer(parameters: SoftmaxParameters(
      name: "prob",
      dependencies: ["pool10"]
      )))

    let networkFile = NSBundle.mainBundle().pathForResource("squeezenet", ofType: "espressomodel")
    network.importFromFile(networkFile!, engine: engine)
  }

//  override func viewWillDisappear(animated: Bool) {
//    network = nil
//  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }
  
  
}

