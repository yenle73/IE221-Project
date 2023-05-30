from st_pages import Page, Section, add_page_title, show_pages

add_page_title()

show_pages([
    Page("pages/main.py", "Home Page", ":house:"),
    Page("pages/page1.py", "Content Based Filtering", ":notebook:"),
    Page("pages/page2.py", "Sentiment and Genre Recommendation", ":blue_book:"),
    Page("pages/page3.py", "Collaborative Filtering", ":bar_chart:"),
])