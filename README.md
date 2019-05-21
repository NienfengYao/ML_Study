# Machine Learning Study
* VGG-16
  * [vgg16_introduction.ipynb](vgg16/vgg16_introduction.ipynb)
* MobileNet
  * [mobilenet_study.ipynb](mobilenet/mobilenet_study.ipynb)
* [onnx_list_layer_operation.ipynb](./onnx_list_layer_operation.ipynb)
  * Show the layer operateion of each model in onnx zoo.
---
* Basic of TensorFlow
  * [Introduction.ipynb](./Introduction.ipynb)
  * [tensorflow_tips_01.ipynb](./tensorflow_tips_01.ipynb)
    * What is difference between tf.truncated_normal and tf.random_normal?
    * tf.nn.conv2d-是怎麼實現卷積的？
    * tf.nn.max_pool 實現池化操作

* Simple Classification Example
  * [simple_classification.ipynb](simple_classification.ipynb)

* Simple Regression Example
  * [simple_regression.ipynb](simple_regression.ipynb)

* TensorBoard Example
  * [tensorboard_ex01.ipynb](tensorboard_ex01.ipynb) 
  * [tensorboard_ex02.ipynb](tensorboard_ex02.ipynb)
    * Graph Visualization of simple_classification.ipynb

* MNIST with Softmax Regression
  * [mnist_softmax_regression.ipynb](mnist_softmax_regression.ipynb)
    * Get data from MNIST and only with softmax regression, no CNN.
  * ToDo
    * [Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
    * [Cross-Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
      * [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
    * [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)

* MNIST with vanilla CNN
  * [mnist_vanilla_cnn.ipynb](mnist_vanilla_cnn.ipynb)
    * Extend mnist_softmax_regression.ipynb, we improve it with a simple CNN 
      * 1 x Input layer.
      * 2 x Convolution layer. (Convolution + Pooling)
      * 1 x Full connected layer.
      * 1 x Ouptut layer.



# Temp
* [Tensorflow-101](https://github.com/c1mone/Tensorflow-101)
* [Tutorials use tf.keras]
