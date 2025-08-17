#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained Keras model
model = load_model('best_model_base.h5')

# Class labels (adjust to your dataset)
species = ['COVID', 'No COVID', 'PNEMONIA']

# App title
st.title("🦠 Covid Classification")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image of a Chest Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image (adjust size to model's expected input)
    img_size = (256, 256)  # change to the size your model was trained on
    image = image.resize(img_size)
    img_array = img_to_array(image) / 255.0  # normalize if trained that way
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]
        st.success(f"Predicted Classification: **{species[pred_class]}**")


# In[ ]:




