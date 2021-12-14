#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import streamlit as st
#import cv2
from PIL import Image, ImageOps
import numpy as np


# In[5]:


model = tf.keras.models.load_model('my_model.hdf5')

st.text('Artist Predictions')

st.write('This is a simple web app to predict the artist of a piece of art')
file = st.file_uploader("Upload your image here")


# In[ ]:


def import_and_predict(image_data,model):
    size = (124,124)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    #img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_resize = image.resize((75,75),resample=PIL.Image.BICUBIC)/255
    img_reshape = img_resize[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("You have not uploaded an image")
else:
    image = Image.open(file).convert('LA')
    st.image(image, use_column_width = True)
    prediction = import_and_predict(image,model)

    if np.argmax(prediction) == 0:
        st.write("Dali")
    elif np.argmax(prediction) == 1:
        st.write("Gogh")
    elif np.argmax(prediction) == 2:
        st.write("Khalo")
    elif np.argmax(prediction) == 3:
        st.write("Monet")
    elif np.argmax(prediction) == 4:
        st.write("Orozco")
    else:
        st.write("Pollock")

    st.text("Probabilities")
    st.write(prediction)
