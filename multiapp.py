import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            'NE YAPILMASINI İSTERSİNİZ:',
            self.apps,
            format_func=lambda app: app['title'])
        app['function']()


    """ MultiApp
    Streamlit de dinamik sayfalama olmadığı için  MultiApp classı ile 
    oluşturulan bu yapı sayesinde selectbox seçimine göre geçiş yapmaktadır."""

