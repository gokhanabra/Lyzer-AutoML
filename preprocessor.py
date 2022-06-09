import numpy as np
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from _Layout import layout
from myScripts import read_dataset, dfYaz
#from sklearn.preprocessing import OneHotEncoder


def app():
    st.title("Preprocessing Islemleri")
    layout()

    # Dataset upload
    upDataSet = st.sidebar.file_uploader(label='Veri setini buraya yukleyiniz', type='csv')

    if upDataSet != None:
        df = read_dataset(upDataSet)
        columns = list(df.columns)
        dfYaz(df)
        with st.sidebar:
            st.write("İstenen kolonları seçiniz:")
            selected = list()

            for i in range(len(columns)):
                sec = st.checkbox(f' {columns[i]}', False)
                if sec:
                    selected.append(i)

            st.write(f'Secilenler: {selected}')
            st.write(selected)

        # with icerisindeki kodu bir ust katmanda islenebilmekte
        if len(selected)>0:
            st.subheader('Kullanilacak olan kolonlar:')
            df2 = pd.DataFrame(df.iloc[:, selected])
            dfYaz(df2)

            col1, col2 = st.columns(2)
            with col1:
                st.write("DataFramede bulunan toplam NaN veri : ", f'{df2.isnull().sum().sum()}')
                columns2 = list(df2.columns)

                for i in range(len(columns2)):
                    if (df2.iloc[:, i].isnull().sum() != 0):
                        st.write(f'{columns2[i]} :  {df2.iloc[:, i].isnull().sum()} adet kayip veri bulunmaktadir.')



            # Bundan sonraki kisimlarda yapilacak olan islem her bir kolon icin ayri ayri kayip veri islemi uygulaticak eger silme secenegi+
            # + varsa tum satir silinmeli. kategorik veriye gore islmei arasitir



            with col2:
                yap = ['Sil', 'Mean', 'Most Frequent', 'Median']
                input = st.radio('Yapilmak isteneni seciniz', yap)

            st.write(input, "Secildi")

            if input == 'Sil':
                df3 = pd.DataFrame(df2.dropna())

            elif input == 'Mean':
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                df3 = pd.DataFrame(imp.fit_transform(df2), columns=df2.columns)

            elif input == 'Most Frequent':
                imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                df3 = pd.DataFrame(imp.fit_transform(df2), columns=df2.columns)

            elif input == 'Median':
                imp = SimpleImputer(missing_values=np.nan, strategy='median')
                df3 = pd.DataFrame(imp.fit_transform(df2), columns=df2.columns)

            dfYaz(df3)

            #Kayip verileri isledikten sonra kategorik veriyi numrik veriye cevirmeli. Modelling kismina csv gosterildiginde tum degerlerin
            #numerik veri olmasi gerekmektedir.

            csv = df3.to_csv(index=False)
            st.download_button(
                label="CSV olarak indir",
                data=csv,
                file_name='OutputPrepCSV.csv',
                mime='text/csv',
            )
    else:
        st.warning('Henuz bir veri seti yuklenmedi!')
