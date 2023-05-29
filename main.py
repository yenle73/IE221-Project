import streamlit as st
from PIL import Image

st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h21>", unsafe_allow_html=True)
st.write(':star: Discover New Movies :star:')
st.markdown("<h4 style='text-align: center; color: #EBB02D;'>By Yen Le</h4>", unsafe_allow_html=True)
image = Image.open('bg.png')
st.image(image)