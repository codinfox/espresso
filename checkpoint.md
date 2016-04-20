---
layout: default
---

<style>
sup:before { content: "["; }
sup:after { content: "]"; }
</style>

<a href="https://github.com/codinfox/espresso" title="Fork me on Github" class="github-corner"><svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: fixed; top: 0; border: 0; right: 0;"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a><style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>

# Checkpoint Report <i class="fa fa-coffee"></i>

**[Zhihao Li](http://codinfox.github.io/)** (zhihaol) and **[Zhenrui Zhang](http://jerryzh168.github.io/)** (zhenruiz)

### Process Review
By far we have implemented a naive CPU and GPU version for the forward path of the neuro network framework. In the first week, we designed the framework according to the achitecture of Caffe and learned Swift programming language and Metal framework. In the second week, we implemented the forward CPU version and tested a few layers. Currently we are implementing the GPU version.

### Goals and Deliverables
We started with the goal of training a small neural network using the framework, however, as we learn more about the constraint we have running the framework on mobile devices and the reality of huge memory consumption of common neural networks, we decided to focus on running a trained model rather than actually training them. We will compare the running time, memory cost and energy consumption of the CPU and GPU version in MNIST.
For the next half of the project, we will explore the possibilities to run a compressed model in our framework. Also, we'll try to support the caffe format models.
In the parallelism competition, we plan to compare the cost of CPU and GPU implementations and run the compressed AlexNet on our framework.

### Revised Schedule


|   Time    | What we plan to do | Status |
|:---------:|:-------------------|:-----:|
| April 1 ~ April 7  | Revise proposal, study the design and architecture of Caffe, learn Swift language and Metal API, implement a simple App for testing, design interfaces for espresso | DONE |
| April 8 ~ April 14  | Develop and test the CPU version | Finished development, need more thorough testing   |
| April 15 ~ April 19 | Develop and test the GPU version | Finished development, need testing    |
| April 20 ~ April 22 | More testing on CPU version and refactor to a better CPU version using Accelerate framework | |
| April 23 ~ April 25 | Test the GPU version | |
| April 26 ~ April 28 | <del>Train neural networks on espresso</del> <br> Run MNIST network(and test our implementations) | |
| April 29 ~ May 3 | Compare the CPU and GPU implementations | |
| May 4 ~ May 6    | <del>Study neural network compression and try to run compressed model</del> <br>Run a compressed model trained by Caffe or other common frameworks |   |
| May 7 ~ Parallel Competition Day | Write final report and prepare for presentation     |    |

----