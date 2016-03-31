---
layout: default
---

<style>
sup:before { content: "["; }
sup:after { content: "]"; }
</style>
# Espresso <i class="fa fa-coffee"></i>

## A minimal high performance parallel neural network framework running on iOS

**[Zhihao Li](http://codinfox.github.io/)** (zhihaol) and **[Zhenrui Zhang](http://jerryzh168.github.io/)** (zhenruiz)

> We are in need of an apple developer account to test the framework on real devices.


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

The demo we plan to do at the parallism competition is ...(Specifically, what will you show us that will demonstrate you did a good job?)

#### HOPE TO ACHIEVE

If we are ahead of schedule, we plan to

### Platform Choice

OSX and iOS. Metal.
<img src="{{ site.baseurl }}/images/hello.svg" alt="sample image">

Thanks to [Shu Uesengi](https://github.com/chibicode) for inspiring and providing the base for this template with his excellent work, [solo](https://github.com/chibicode).

### Schedule

|   Time    | What we plan to do | What we actually did  |
|:---------:|:-------------------|:-----:|
| April 1   | Revise proposal, study the design and architecture of Caffe, learn Swift language and Metal API      |  |
| April 7   | Design interfaces for espresso, implement a simple app for testing and an initial implementation of the cpu version of espresso framework           |    |
| April 15  | Develop the CPU version and test the framework  |     |
| April 22  | Develop and test GPU version |  |
| May 1     | Debug the framework, prepare demo |   |
| May 7 	| Wrap up, Final report and Prepare for presentation     |    $1 |

**Share the excitement with your friends by**
{% include sharing.html %}

----

##### References:

[^1]: Kim, Yong-Deok, et al. "Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications." *arXiv preprint arXiv:1511.06530* (2015).

[^2]: Han, Song, Huizi Mao, and William J. Dally. "A deep neural network compression pipeline: Pruning, quantization, huffman encoding." *arXiv preprint arXiv:1510.00149* (2015).

[^3]: Chen, Wenlin, et al. "Compressing neural networks with the hashing trick." *arXiv preprint arXiv:1504.04788* (2015).

[^4]: Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Low precision arithmetic for deep learning." *arXiv preprint arXiv:1412.7024* (2014).

[^5]: Gupta, Suyog, et al. "Deep learning with limited numerical precision." *arXiv preprint arXiv:1502.02551* (2015).

[^6]: Huberty, K., Lipacis, C. M., Holt, A., Gelblum, E., Devitt, S., Swinburne, B., ... & Chen, G. (2011). Tablet Demand and Disruption. *Tablet*.

[^7]: Jia, Yangqing, et al. "Caffe: Convolutional architecture for fast feature embedding." *Proceedings of the ACM International Conference on Multimedia. ACM*, 2014.
