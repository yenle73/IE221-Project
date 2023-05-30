from st_pages import Page, Section, add_page_title, show_pages
import streamlit as st
from PIL import Image

st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #10316B;'>&#10024 Discover New Movies &#10024</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #EBB02D;'>By Yen Le</h4>", unsafe_allow_html=True)
image = Image.open('bg.png')
st.image(image)

#add_page_title()

show_pages([
    Page("main.py", "Home Page", ":house:"),
    Page("pages/page1.py", "Content Based Filtering", ":notebook:"),
    Page("pages/page2.py", "Sentiment and Genre Recommendation", ":blue_book:"),
    Page("pages/page3.py", "Collaborative Filtering", ":bar_chart:"),
])