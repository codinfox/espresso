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


### Background

Good clean read is set up with readability first in mind. Whatever you want to communicate here can be read easily, and without distraction. Of course, it's fully responsive, which means people can read it naturally on any phone, or tablet. Write it in markdown in <code>index.md</code> and get a beautifully published piece.

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

> "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

### The Challenge

The task of training and running neural networks on a iOS device is itself challenging.

* **Memory Limitation** The latest version of iPhone (iPhone 6S) has only 2 GB RAM. This makes running a network on such device very difficult, not to mention training on it. To compensate this issue, we may take advantage of recent research outgrowth on compression of deep neural networks [^1] [^2] [^3] or use low-precision networks [^4] [^5].

### Resources

Simply add the following line anywhere in your markdown:

<pre><code>{% raw  %}
{% include sharing.html %}
{% endraw %}
</code></pre>

and get a nice responsive sharing ribbon.


Add this at the bottom, or the top, or between every other paragraph if you're desprate for social validation.

Just remember to customize the buttons to fit your url in the `_includes/sharing.html` file. These buttons are made available and customizable by the good folks at kni-labs. See the documentation at [https://github.com/kni-labs/rrssb](https://github.com/kni-labs/rrssb) for more information.

### Goals and Deliverables

<i class="fa fa-quote-left fa-3x fa-pull-left fa-border"></i> Now you can use all the cool icons you want! [Font Awesome](http://fontawesome.io) is indeed awesome. But wait, you don't need this sweetness and you don't want that little bit of load time from the font awesome css? No problem, just disable it in the `config.yml` file, and it won't be loaded.

<ul class="fa-ul">
  <li><i class="fa-li fa fa-check-square"></i>you can make lists...</li>
  <li><i class="fa-li fa fa-check-square-o"></i>with cool icons like this,</li>
  <li><i class="fa-li fa fa-spinner fa-spin"></i>even ones that move!</li>
</ul>

If you need them, you can stick any of the [605 icons](http://fontawesome.io/icons/) anywhere, with any size you like. ([See documentation](http://fontawesome.io/examples/))

<i class="fa fa-building"></i>&nbsp;&nbsp;<i class="fa fa-bus fa-lg"></i>&nbsp;&nbsp;<i class="fa fa-cube fa-2x"></i>&nbsp;&nbsp;<i class="fa fa-paper-plane fa-3x"></i>&nbsp;&nbsp;<i class="fa fa-camera-retro fa-4x">

### Platform Choice

Images play nicely with this template as well. Add diagrams or charts to make your point, and the template will fit them in appropriately.

<img src="{{ site.baseurl }}/images/hello.svg" alt="sample image">

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Thanks to [Shu Uesengi](https://github.com/chibicode) for inspiring and providing the base for this template with his excellent work, [solo](https://github.com/chibicode).


### Schedule


**Share the excitement with your friends by**
{% include sharing.html %}

----

##### Footnotes:

[^1]: Kim, Yong-Deok, et al. "Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications." *arXiv preprint arXiv:1511.06530* (2015).

[^2]: Han, Song, Huizi Mao, and William J. Dally. "A deep neural network compression pipeline: Pruning, quantization, huffman encoding." *arXiv preprint arXiv:1510.00149* (2015).

[^3]: Chen, Wenlin, et al. "Compressing neural networks with the hashing trick." *arXiv preprint arXiv:1504.04788* (2015).

[^4]: Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Low precision arithmetic for deep learning." *arXiv preprint arXiv:1412.7024* (2014).

[^5]: Gupta, Suyog, et al. "Deep learning with limited numerical precision." *arXiv preprint arXiv:1502.02551* (2015).
