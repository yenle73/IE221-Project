from library import library as lib
from st_pages import Page, add_page_title, show_pages

show_pages(
    [
        Page("pages/main.py", "Home", "🏠"),
        Page("pages/page1.py", "Content Based Recommnder", ":books:"),
        Page("pages/page2.py", "Mood and Genre Based Recommnder", "📖"),
        Page("pages/page3.py", "Collaborative Recommender", "✏️"),
    ]
)

