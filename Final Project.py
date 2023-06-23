import streamlit as st
import streamlit as st
import pygame
import numpy as np
from PIL import Image
import os
import pandas as pd
import base64
import io


# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            color: white;  /* Set the text color to white */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
add_bg_from_local('./images/jvl.png')
st.write(" ")
st.write(" ")
st.write(" ")

# Set the layout and content
col1, col2, col3 = st.columns([1., 20, 1])
with col2:
    st.title('LeBron James vs Michael Jordan')
    st.markdown("<h1 style='text-align: center; font-size: 25px; color: black;'>Who is the GOAT?</h1>", unsafe_allow_html=True)
    st.markdown("")  # add 3 line breaks

col1, col2, col3 = st.columns([4, 17, 1])
with col2:
    st.markdown("<h1 style='font-size: 20px; color: black;'>Presented by: Dr. Seyed Mohsen Jebreiil Khadem</h1>", unsafe_allow_html=True)
   # st.markdown("<h1 style='font-size: 20px; color: black;'>Teachers: Diana, Marija, Jens, Sara, Carmine, Carlos</h1>", unsafe_allow_html=True)
    st.markdown("")  # add 3 line breaks

col1, col2, col3 = st.columns([1.25, 1, 1])
with col2:
    st.markdown("<h1 style='font-size: 20px; color: black;'>Date: 2023-05-19</h1>", unsafe_allow_html=True)

# # Load and display the logos
# logo_1 = Image.open("./images/spiced.jpg").resize((140, 140))
# logo_2 = Image.open("./images/logo_2.png").resize((140, 120))

# # Create containers for the logos
# logo_container1 = st.sidebar.container()
# logo_container2 = st.sidebar.container()

# # Display the logos in the containers
# with logo_container1:
#     st.image(logo_1, use_column_width=False)

# with logo_container2:
#     st.image(logo_2, use_column_width=False)
st.write(" ")   
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 
st.write(" ") 

st.write('fadeawayworld.net')