//
//  FirstViewController.swift
//  EspressoHostApp
//
//  Created by Zhihao Li on 5/5/16.
//  Copyright © 2016 CMU. All rights reserved.
//

import UIKit

class FirstViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var label: UILabel!
  @IBAction func recognize(sender: AnyObject) {
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
    picker.dismissViewControllerAnimated(true, completion: nil)
  }

  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view, typically from a nib.
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }


}

