# TensorFlow_Study
* Environment
  * TensorFlow: 1.13.0
  * Python: 3.6.6

* Basic of TensorFlow
  * Introduction.ipynb
  * tensorflow_tips_01.ipynb
    * What is difference between tf.truncated_normal and tf.random_normal?

* Simple Classification Example
  * simple_classification.ipynb

* Simple Regression Example
  * simple_regression.ipynb

* TensorBoard Example
  * tensorboard_ex01.ipynb 
  * tensorboard_ex02.ipynb: Graph Visualization of simple_classification.ipynb

* MNIST with Softmax Regression
  * mnist_softmax_regression.ipynb 
    * Get data from MNIST and only with softmax regression, no CNN.
  * ToDo
    * [Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
    * [Cross-Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
      * [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
    * [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)

* MNIST with vanilla CNN
  * mnist_vanilla_cnn.ipynb
    * Extend mnist_softmax_regression.ipynb, we improve it with a simple CNN 
      * 1 x Input layer.
      * 2 x Convolution layer. (Convolution + Pooling)
      * 1 x Full connected layer.
      * 1 x Ouptut layer.
  * ToDo
    * tf.truncated_normal
    * tf.nn.dropout
    * tf.nn.conv2d
    * tf.nn.max_pool


# Temp
* [Tensorflow-101](https://github.com/c1mone/Tensorflow-101)
