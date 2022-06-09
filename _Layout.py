import streamlit as st
from extra_html import html

def layout():
    #Ana Baslik ve logonun acilista goruntulenmesi
    st.markdown(html(), unsafe_allow_html=True)
    st.write("Lyzer Data Knowladge Tool")

    st.title('Veri Seti Islemleri')
    st.subheader('Veri Seti Bilgileri: ')

