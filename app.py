import gradio as gr
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = tf.keras.models.load_model('Trained_Model.h5')

# Define the class labels
class_labels = {
    0: 'Glioma',
    1: 'Meningioma',
    3: 'Pituitary',
    2: 'No tumor'
}

# Create the image generator for preprocessing
img_gen = ImageDataGenerator(rescale=1./255)

# Define the function to predict tumor type
def predict_image(file):
    # Load the image or video
    cap = cv2.VideoCapture(file.name)
    if cap.isOpened():
        ret, frame = cap.read()
        # Check if it's an image or video
        if frame is not None:
            # Preprocess the image            
            frame = cv2.resize(frame, (150, 150))
            frame = np.expand_dims(frame, axis=-1)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype('float32')
            frame = img_gen.standardize(frame)

            # Predict the tumor type
            prediction = model.predict(frame)
            label = class_labels[np.argmax(prediction)]
        else:
            label = "No frames found in the video"
    else:
        label = "Could not open the file"
    return label

# Create the Gradio interface
input_type = gr.inputs.File(label="Upload an image or video to predict the brain tumor type")
output_type = gr.outputs.Textbox(label="Predicted Brain Tumor")
title = "Brain Tumor Classification"
description = "Upload an MRI scan image or video to predict the corresponding brain tumor type"
iface = gr.Interface(fn=predict_image, inputs=input_type, outputs=output_type, title=title, description=description)
if __name__ == '__main__':
    iface.launch(inline=False)