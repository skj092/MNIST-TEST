# Streamlit Handwritten Digit Recognition App

import streamlit as st
import numpy as np
import cv2
from model import LeNet
from CFG import load_checkpoint, device
import CFG
# import streamlit canvas 
from streamlit_drawable_canvas import st_canvas
import torch 

# Title
st.title("Handwritten Digit Recognition")
st.header("Draw a digit and let the model predict it!")

# Canvas
canvas_result = st.empty()
canvas = st.empty()
canvas = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)
# Predict the output of canvas image 
def predict(img):
    # load model
    model = LeNet()
    load_checkpoint(torch.load(CFG.CHECKPOINT_FILE), model)
    model.to(device)
    model.eval()
    # loading image
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('image shape', img.shape)
    img = img.reshape(1, 1, 28, 28)
    img = torch.from_numpy(img).float()
    # predict
    output = model(img)
    _, pred = torch.max(output, 1)
    return pred.item()
    

# Predict button
if st.button("Predict", key="predict"):
    result = predict(canvas.image_data)
    canvas_result.image(canvas.image_data)
    st.write("The model predicted: ", result)

# loading the pytorch checkpoint model
model = LeNet()
load_checkpoint(torch.load(CFG.CHECKPOINT_FILE), model)
print('model loaded successfully')

# clear canvas button
if st.button("Clear", key="clear"):
    canvas.clear()

