# vgg16_final

* Base the study of privious vgg16 examples, I want to redo the predict/training/transfer learning with single VGG16 modle.
  * [vgg16_pretrained_predict.ipynb](vgg16/vgg16_pretrained_predict.ipynb)
  * [vgg-cifar10_train.ipynb](vgg16/vgg-cifar10_train.ipynb)
  * [Transfer_Learning.ipynb](vgg16/transfer-learning/Transfer_Learning.ipynb)

* Example: Predict
  * [vgg16f_trained_predict.ipynb](vgg16f_trained_predict.ipynb)
    * Add function: predict multiple pictures

* Example: Training with CIFAR-10 dataset
  * [vgg16f_training.ipynb](vgg16f_training.ipynb)
    * Because conv1_1's input feature is 3, so we have to change the input image 
      shape (3, 32, 32) in RGB to shape (32, 32, 3) in GBR

  * vgg16_cifar10.py
    * Base the original VGG16 mode, change the last 1000 output to 10 output.
    * Add load weight function, not auto-load.
    * Add setting of mean value.

* functions
  * The missed files, please download from [VGG in TensorFlow](https://www.cs.toronto.edu/~frossard/post/vgg16/)
  * example: trained vgg16 to predict. (Done)
  * example: vgg16 training fully with cifar-10 data (Failed)
    * Because the original VGG16 model is fixed in 1000 output.
      If we want to train with CIFAR-10 dataset which is 10 output, we have to modify the original network.
      It's not what we want. (We want to base the original VGG16 network to do test)
    * Going:
      * Base the original VGG16 mode, only change the last output layer to create vgg16_cifar10 class to do the exercise.
      * Issue: training failed, we can't get good accuracy.
  * example: vgg16 transfer learning with flower dataset. (Done)
