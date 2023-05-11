

import numpy as np
import cv2
import tensorflow as tf
import gradio as gr
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout

model = tf.keras.models.load_model('/kaggle/input/brain-tumor-model/braintumor3.h5')

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    label_idx = np.argmax(prediction)
    return labels[label_idx]

input_interface = gr.inputs.Image(shape=(150, 150))
output_interface = gr.outputs.Label(num_top_classes=1)

grapp = gr.Interface(fn=predict, inputs=input_interface, outputs=output_interface, title='Brain Tumor Detection', description='Detect brain tumors from MRI images')

if __name__ == '__main__':
    grapp.launch(inline = False)