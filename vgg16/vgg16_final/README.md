# vgg16_final

* Base the privious vgg16 study, I exercise the example: vgg16_final
  * [vgg16_pretrained_predict.ipynb](vgg16/vgg16_pretrained_predict.ipynb)
  * [vgg-cifar10_train.ipynb](vgg16/vgg-cifar10_train.ipynb)
  * [Transfer_Learning.ipynb](vgg16/transfer-learning/Transfer_Learning.ipynb)

* The difference with previous is
  * The previous studies doesn't use the same vgg16 model.
  * I want to use the same vgg16 model to re-do these exercise.
  
* functions
  * The uncommited files, please download from [VGG in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)
  * vgg16.py
    * I modify it for 
      * Add load weight function, not auto-load
      * Add function: predict multiple pictures
  * example: trained vgg16 to predict
  * example: vgg16 training fully with cifar-10 data
  * example: vgg16 transfer learning with cifar-10 data
