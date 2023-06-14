from library import library as lib

lib.st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
lib.st.markdown("<h3 style='text-align: center; color: #10316B;'>&#10024 Discover New Movies &#10024</h3>", unsafe_allow_html=True)
lib.st.markdown("<h5 style='text-align: center; color: #EBB02D;'>By Le Thi Kim Yen</h5>", unsafe_allow_html=True)
lib.st.markdown("<h5 style='text-align: center; color: #EBB02D;'>Student ID: 21521695</h5>", unsafe_allow_html=True)
image = lib.Image.open('bg.png')
lib.st.image(image)


#add_page_title()

lib.show_pages([
    lib.Page("main.py", "Home Page", ":house:"),
    lib.Page("pages/page1.py", "Content Based Recommender", ":notebook:"),
    lib.Page("pages/page2.py", "Mood and Genre Based Recommender", ":blue_book:"),
    lib.Page("pages/page3.py", "Collaborative Recommender", ":bar_chart:"),
])