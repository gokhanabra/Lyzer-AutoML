import pandas as pd
import streamlit as st
import seaborn as sns



#Datasetin Okunması
def read_dataset(x):
    return pd.read_csv(x)

#Streamlit uzerinden df yazdirma
def dfYaz(df):
    return st.dataframe(df)

#Veri setinin ayrilmasi
def ayir(df,x1,x2,y):
    X = df.iloc[:, int(x1): int(x2)].values
    Y = df.iloc[:, : int(y)].values




def inspect_column(df):
    columns = list(df.columns)
    return columns

def df_kaydet(df):
    df.to_csv("yeniDataSet.csv")

def pairplot(df):
    sns.set_theme(style="darkgrid")
    sns.pairplot(df, hue="Pregnancies")

