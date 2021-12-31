# using-pretrained-network

Neural network classification with pretrained convolution base

### Prequisites

* numpy
* matplotlib
* glob
* seaborn
* tensorflow
* sklearn

### Dataset

I have used fruits Kaggle dataset which contains multiple types of fruit photos from different angles. The link to the dataset can be seen below:

https://www.kaggle.com/moltean/fruits

### Models

* VGG16
* ResNet50
* Xception

### Results

Validation accuracies:

* VGG16: 0.9435
* ResNet50: 0.6155
* Xception: 0.9137

It can clearly be seen that VGG16 and Xception architectures were superior against the ResNet architecture with this dataset.
