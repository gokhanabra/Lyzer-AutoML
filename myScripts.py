import pandas as pd
import streamlit as st

#Datasetin Okunması
def read_dataset(x):
    return pd.read_csv(x)

#Streamlit üzerinden df yazdırma
def dfYaz(df):
    return st.dataframe(df)

#Veri setinin ayrılmasi
def ayir(df,x1,x2,y):
    X = df.iloc[:, int(x1): int(x2)].values
    Y = df.iloc[:, : int(y)].values

#Numerik olmayan kolonların tespiti
def no_numeric_col(df):
    num_cols = df._get_numeric_data().columns
    noNum = list(set(df.columns) - set(num_cols))
    return noNum

#Kolonların çekilmesi
def inspect_column(df):
    columns = list(df.columns)
    return columns

#Data Frame kayıt etme
def df_kaydet(df):
    df.to_csv("yeniDataSet.csv")

#Hangi sınfılandırmanın hiper parametresi olduunu getirir
def hyp_header(clf_name):
    st.subheader("Hyperparameter Tuning for "+f'{clf_name}'+" Classification")

#Kullanıcıya verisetindekı mııs value ve non numerik kolonların bilgisini erana yazar.
def Message(miss_value, noNum):
    st.write("*DataFrame de toplam", f'{miss_value}' " adet kayıp veri bulunmaktadır.*")
    if len(noNum) > 0:
        st.write("*DataFramede", f'{noNum}' " kolonunda numerik olmayan veri bulunmaktadır.*")
    else:
        st.write("*DataFrame de numerik olamayan veri bulunmamakadır.*")

"""
    Bazı işlemleri yapabilmek için kullanılabilir bir script sayfası oluşturulmuştur.
"""