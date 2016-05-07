---
layout: default
---

<style>
sup:before { content: "["; }
sup:after { content: "]"; }
</style>

<a href="https://github.com/codinfox/espresso" title="Fork me on Github" class="github-corner"><svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: fixed; top: 0; border: 0; right: 0;"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a><style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>

# Espresso <i class="fa fa-coffee"></i>

## A minimal high performance parallel neural network framework running on iOS

**[Zhihao Li](http://codinfox.github.io/)** (zhihaol) and **[Zhenrui Zhang](http://jerryzh168.github.io/)** (zhenruiz)

<a href="http://codinfox.github.io/espresso/checkpoint" target="_blank">Checkpoint Report</a>

### Background

According to Morgan Stanley Research, as of the year of 2011, half of the computing devices worldwide are mobile devices [^6]. The intelligent mobile applications are changing people's lives. However a quite thorough survey, we find no fully functional deep neural network framework on iOS. Therefore, we want to implement our own.

This framework features ***well designed and easy to use API***, and ***high performance parallel neural network implementation*** based on Metal.

With such framework, software engineers can easily train and test network on their iOS devices. This can potentially lead to many interesting applications. For example, an application that can recognize daily objects ***in real time without connection to internet***. Or photo collection applications that can recognize all your friends based on ***personalized fine-tuning without any threat to privacy***.

We envision a great future market opening for such framework.

### The Challenge

The task of training and running neural networks on a iOS device is itself challenging.

* **Memory Limitation** The latest version of iPhone (iPhone 6S) has only 2 GB RAM. This makes running a network on such device very difficult, not to mention training on it. To compensate this issue, we may take advantage of recent research outgrowth on compression of deep neural networks [^1] [^2] [^3] or use low-precision networks [^4] [^5].
* **High Performance Computing** Parallelizing a neural network implementation on iOS devices is an unprecedented task. We will explore the possibility of Metal API to implement a GPGPU version of the framework.
* **Learning of a New Language** Both of us are not familiar with Swift 2, the programming language. This would be a challenge for us in the early stage of implementation.

Bearing so many challenges, this project is still promising. The task of training and testing neural networks is highly parallizable, as the computation inside a layer is independent (we currently don't support intra-layer connections). The locality should be good as weights within the same layer should be stores adjacently. And typically there is not much divergence in the network training and testing - all the weights are updated at the same time within a layer.



### Resources

We will start the project from scratch. The framework will be mainly running on iOS devices with limited support to OSX devices. We will use the high-level architecture of Caffe [^7] as our reference.

***We are in need of a apple developer account to test the framework on real devices.***

### Goals and Deliverables

#### PLAN TO ACHIEVE
In this project, we want to develop a Caffe-like deep neural network framework running on iOS/OSX devices, in both CPU and GPU, that provides usable primitives to

* Define a neural netowrk
* Train a small neural network
* Run compressed models

To achieve this, we will implement

* **Layers**
	* `ImageData` layer
	* `Convolution` layer
	* `ReLU` layer
	* `FullyConnected` layer
	* `Softmax` layer: as output layer, no BP needed
	* `SoftmaxWithLoss` layer
	* `Pooling` layer: max pooling and average pooling
	* `Dropout` layer
	* `LRN` layer
* **Optimizer**
	* `SGDOptimizer`

We want our system to be usable in mobile devices, therefore, the performance goal would be to have a user acceptable memory, energy, computation cost and response time to train on a reasonably sized dataset, and to run a compressed model.

#### HOPE TO ACHIEVE

If we are ahead of schedule, we plan to port some other layers and optimizers from Caffe to our framework.

#### Demo
We will be demonstrating an application developed based on our framework. It could be a application to recognize things. Also, we will be comparing the CPU implementation and GPU implementation in terms of speedup and energy consumption.

### Platform Choice

OSX and iOS based on Metal framework.

### Schedule

|   Time    | What we plan to do | Status |
|:---------:|:-------------------|:-----:|
| April 1 ~ April 7  | Revise proposal, study the design and architecture of Caffe, learn Swift language and Metal API, implement a simple App for testing, design interfaces for espresso | DONE |
| April 8 ~ April 14  | Develop and test the CPU version | Finished development, need more thorough testing   |
| April 15 ~ April 21 | Develop and test the GPU version | Finished development, need testing    |
| April 22 ~ April 28 | Run MNIST network(and test our implementations) |  |
| April 29 ~ May 5   | Run a compressed model trained by Caffe or other common frameworks |   |
| May 6 ~ Parallel Competition Day | Write final report and prepare for presentation     |    |



##### References:

[^1]: Kim, Yong-Deok, et al. "Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications." *arXiv preprint arXiv:1511.06530* (2015).

[^2]: Han, Song, Huizi Mao, and William J. Dally. "A deep neural network compression pipeline: Pruning, quantization, huffman encoding." *arXiv preprint arXiv:1510.00149* (2015).

[^3]: Chen, Wenlin, et al. "Compressing neural networks with the hashing trick." *arXiv preprint arXiv:1504.04788* (2015).

[^4]: Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Low precision arithmetic for deep learning." *arXiv preprint arXiv:1412.7024* (2014).

[^5]: Gupta, Suyog, et al. "Deep learning with limited numerical precision." *arXiv preprint arXiv:1502.02551* (2015).

[^6]: Huberty, K., Lipacis, C. M., Holt, A., Gelblum, E., Devitt, S., Swinburne, B., ... & Chen, G. (2011). Tablet Demand and Disruption. *Tablet*.

[^7]: Jia, Yangqing, et al. "Caffe: Convolutional architecture for fast feature embedding." *Proceedings of the ACM International Conference on Multimedia. ACM*, 2014.
