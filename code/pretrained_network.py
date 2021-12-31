from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from glob import glob
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

train_path = "/home/kerem/Desktop/fruits-360/fruits-smaller-dataset/Training/"
test_path = "/home/kerem/Desktop/fruits-360/fruits-smaller-dataset/Test/"

base_dir = "/home/kerem/Desktop/fruits-360/fruits-smaller-dataset"
train_dir = os.path.join(base_dir,"Training")
test_dir = os.path.join(base_dir,"Test")

image_files = os.listdir(train_dir)

class Pretrained_network():

    def __init__(self, image_size=[100, 100], batch_size=32, epochs=4):

        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs

    def generating_data(self):



        base_dir = "/home/kerem/Desktop/fruits-360/fruits-smaller-dataset"
        self.train_path = os.path.join(base_dir, "Training")
        self.test_path = os.path.join(base_dir, "Test")

        self.image_files = os.listdir(train_dir)

        self.image_files = glob(self.train_path + '/*/*.jp*g')
        self.test_path = glob(self.test_path + '/*/*.jp*g')

        self.folders = glob(self.train_path + '/*')

        self.train_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2,
            preprocessing_function=preprocess_input
        )

        self.validation_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        self.train_generator = self.train_gen.flow_from_directory(self.train_path,
                                                                  target_size=self.image_size,
                                                                  shuffle=True,
                                                                  batch_size=self.batch_size,
                                                                  subset='training',
                                                                  class_mode='categorical')

        self.validation_generator = self.validation_gen.flow_from_directory(self.train_path,
                                                                            target_size=self.image_size,
                                                                            shuffle=False,
                                                                            batch_size=self.batch_size,
                                                                            subset='validation',
                                                                            class_mode='categorical')
        self.test_datagen = ImageDataGenerator(rescale=1 / 255)

        return self.train_generator, self.validation_generator, self.test_datagen

    def convolution_model(self, model):

        convolution_base = model(include_top=False, weights='imagenet', input_shape=self.image_size + [3])

        # dont train existing weights
        for layer in convolution_base.layers:
            layer.trainable = False

        return convolution_base

    def neural_networks(self, convolution_base):

        model = Sequential()
        model.add(convolution_base)
        model.add(Flatten())
        model.add(Dense(1000, activation='relu', input_dim=100 * 100 * 3))
        model.add(Dense(len(self.folders), activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model

    def train_model(self, model, train_generator, validation_generator, test_datagen):

        history = model.fit(train_generator,
                            validation_data=validation_generator,
                            epochs=self.epochs,
                            steps_per_epoch=train_generator.samples // self.batch_size,
                            validation_steps=validation_generator.samples // self.batch_size)

        return history

    def get_confusion_matrix(self, model):
        print("***Generating confusion matrix***")
        predictions = []
        targets = []
        i = 0
        n_images = 0
        for x, y in self.test_datagen.flow_from_directory(self.test_path,
                                                          target_size=self.image_size,
                                                          shuffle=False,
                                                          batch_size=self.batch_size * 2):
            i += 1
            n_images += len(y)
            if i % 50 == 0:
                print(f'{n_images} images processed')
            p = model.predict(x)
            p = np.argmax(p, axis=1)
            y = np.argmax(y, axis=1)
            predictions = np.concatenate((predictions, p))
            targets = np.concatenate((targets, y))
            if len(targets) >= (len(self.image_files)//5):
                break

        self.cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(30, 30))
        sns.heatmap(self.cm, annot=False, cmap='Blues')
        plt.show()

    def accuracy_and_loss(self, history):

        plt.figure(figsize=(5, 5))
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(history.history['loss'], label='loss')
        axs[0, 0].set_title('Loss')
        axs[0, 1].plot(history.history['val_loss'], label='val_loss')
        axs[0, 1].set_title('Validation Loss')
        axs[1, 0].plot(history.history['acc'], label='acc')
        axs[1, 0].set_title('Accuracy')
        axs[1, 1].plot(history.history['val_acc'], label='val_acc')
        axs[1, 1].set_title('Validation Accuracy')


pre = Pretrained_network()

train_generator, validation_generator, test_datagen = pre.generating_data()

VGG16_convolution_model = pre.convolution_model(VGG16)
VGG16_model = pre.neural_networks(VGG16_convolution_model)

resnet_convolution_model = pre.convolution_model(ResNet50)
ResNet50_model = pre.neural_networks(resnet_convolution_model)

Xception_convolution_model = pre.convolution_model(Xception)
Xception_model = pre.neural_networks(Xception_convolution_model)

#** VGG16 **

history_vgg = pre.train_model(VGG16_model, train_generator, validation_generator, test_datagen)

pre.get_confusion_matrix(VGG16_model)

pre.accuracy_and_loss(history_vgg)

#** ResNet50 **

history_res = pre.train_model(ResNet50_model, train_generator, validation_generator, test_datagen)

pre.get_confusion_matrix(ResNet50_model)

pre.accuracy_and_loss(history_res)

#** Xception **

history_xcep = pre.train_model(Xception_model, train_generator, validation_generator, test_datagen)

pre.get_confusion_matrix(Xception_model)

pre.accuracy_and_loss(history_xcep)