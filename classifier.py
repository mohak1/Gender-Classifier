# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:28:54 2018

@author: Mohak

Male - Female Classifier using a CNN
This model tries to classify human faces and lables them as a 'male' or a 'female'
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np

trainingPath = 'G:\\Datasets\man-woman\\Training'

testingPath = 'G:\\Datasets\\man-woman\\Testing'

model = Sequential()

model.add(Conv2D(64,3,3, input_shape=(64,64,3), border_mode='same', activation = 'relu'))
#Convolution layer 1:
#32 feature maps
#3 cols of filter
#3 rows of filter
#size of image is 64x64

model.add(MaxPooling2D(pool_size=(4,4), strides=2))
#after pooling, the size become half, i.e 32x32
#hence input shape in the second convolution layer is 32x32
#we can skip the input_shape parameter as the NN would know what size to expect

#model.add(Conv2D(64, 2, 2, input_shape=(32,32,3), border_mode='same', activation='relu'))
model.add(Conv2D(64, 2, 2, border_mode='same', activation='relu'))
#Convulution layer 2:
#32 feature maps
#2 cols of filter
#2 rows of filter

model.add(MaxPooling2D(pool_size=(3,3), strides=1))

model.add(Flatten())

#Starting the ANN
#we do not have to specify the input size as the NN would already know that
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=16, activation='relu'))
model.add(Dropout(rate=0.4))

model.add(Dense(units=2, activation='softmax'))
#while using softmax, the output_dim or units should be equal to num_classes, which here is 2.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Image Agumentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        trainingPath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        testingPath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        training_set,
        steps_per_epoch=7376,
        epochs=10,
        validation_data=test_set,
        validation_steps=952)

model.save('G:\\Projects\\m-w highEpochs.h5')
model.load_weights('G:\\Projects\\m-w highEpochs.h5')

print(training_set.class_indices)
#prediction
import numpy as np
from keras.preprocessing import image
#img = image.load_img(path='G:\\Datasets\\man-woman\\Testing\\woman\\milly3.jpg', target_size=(64,64))
#img = image.load_img(path='G:\\Datasets\\extra men\\14-05.jpg', target_size=(64,64))
img = image.load_img(path='G:\\dip.jpg', target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
result = model.predict(img)
if result[0][0] >= 0.5:
    print('male')
else:
        print('female')
