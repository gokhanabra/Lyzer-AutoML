import streamlit as st
from extra_html import html

def layout():
    #Ana Baslik ve logonun acilista goruntulenmesi
    st.markdown(html(), unsafe_allow_html=True)
    st.write("Lyzer Data Processing and Modelling Tool")

    st.title('Veri Seti İşlemleri')
    st.subheader('Veri Seti Bilgileri: ')

    """
        Layout fonksiyonunda ekstra HTML fonskiyonu çağırılarak
        Processing ve Modelling aşamalarında sabit değişmez bir yapı
        elde edilerek daha temiz kodlama hedeflenmiştir.
    """