import streamlit as st
import cv2
import numpy as np
import pickle

with open("Egg-image-classification/model.pkl","rb") as file:
    model = pickle.load(file)

st.title("Egg Image Classifier")
st.header("Choose an image file")
file = st.file_uploader("",type=["jpg","jpeg","png"])

img_size = (100,100)
if file is not None:
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img,img_size)
    img_arr = img.flatten()
    img_arr = np.array(img_arr).reshape(1,-1)
    res = model.predict(img_arr)
    st.image(img)
    if(res == 0):
        st.write("It's a Cracked Egg!:broken_heart:")
    elif(res == 1):
        st.write("You fooled me there is no egg in this image. :angry:")
    else:
        st.write("It's a good Egg! :egg:")
    
