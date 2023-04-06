# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Define the input shape
input_shape = (150, 150, 3)

# Define the number of classes
num_classes = 4

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define the paths to the training and test data
train_data_path = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_data_path = '/kaggle/input/brain-tumor-mri-dataset/Testing'

# Create the generators
train_generator = train_datagen.flow_from_directory(train_data_path, 
                                                    target_size=input_shape[:2], 
                                                    batch_size=32, 
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_data_path, 
                                                  target_size=input_shape[:2], 
                                                  batch_size=32, 
                                                  class_mode='categorical')

# Define the ResNet model
resnet_model = tf.keras.applications.ResNet101(include_top=False, 
                                              weights='imagenet', 
                                              input_shape=input_shape)

# Add the classification layers on top of ResNet
classifier = keras.Sequential()
classifier.add(resnet_model)
classifier.add(layers.Flatten())
classifier.add(layers.Dense(256, activation='relu'))
classifier.add(layers.Dropout(0.5))
classifier.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = classifier.fit(train_generator, 
                         steps_per_epoch=train_generator.samples // train_generator.batch_size, 
                         epochs=50, 
                         validation_data=test_generator, 
                         validation_steps=test_generator.samples // test_generator.batch_size)

classifier.save('/kaggle/working/Classifier.h5')

# Evaluate the model
score = classifier.evaluate(test_generator, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])