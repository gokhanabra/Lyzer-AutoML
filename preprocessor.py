import streamlit as st
import pandas as pd
import io
from sklearn import preprocessing
from _Layout import layout
from myScripts import read_dataset, dfYaz
from myScripts import no_numeric_col

def app():
    st.title("Preprocessing İşlemleri")
    layout()

    # Dataset upload
    upDataSet = st.sidebar.file_uploader(label='Veri setini buraya yukleyiniz', type='csv')
    if upDataSet != None:
        df = read_dataset(upDataSet)
        columns = list(df.columns)
        dfYaz(df)
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        st.text(df_info)
        st.write(upDataSet.name)
        #Kullanılmak istenen kolonların seçilmesi
        with st.sidebar:
            st.write("İstenen kolonları seçiniz:")
            selected = list()
            for i in range(len(columns)):
                sec = st.checkbox(f' {columns[i]}', False)
                if sec:
                    selected.append(i)

            st.write(f'Secilenler: {selected}')
            st.write(selected)

        #Seçilen kolonların dinamik olarak gösterilmesi
        if len(selected) > 0:
            st.subheader('Kullanilacak olan kolonlar:')
            df2 = pd.DataFrame(df.iloc[:, selected])
            dfYaz(df2)

            st.write("DataFramede bulunan toplam NaN veri : ", f'{df2.isnull().sum().sum()}')
            yap_na = ['Sil', 'Mean', 'Most Frequent', 'Median']
            yap_en = ['Label Encoder', 'One Hot Encoder', 'Satırı Sil']
            noNum = no_numeric_col(df)
            le = preprocessing.LabelEncoder()

            """Kolon kolon tespit edilme"""

            #Kategorik verinin olması durumu
            for i in range(len(selected)):
                columnName = columns[selected[i]]
                if noNum.count(columnName) != 0:
                    input_ohe = st.radio(f'{columnName} '"Kolonunda kategorik veri  için yapilmak isteneni seçiniz", yap_en, key=selected[i])
                    if input_ohe == 'Label Encoder':
                        df2[columnName] = le.fit_transform(df2[columnName])
                    elif input_ohe == 'One Hot Encoder':
                        ohe_df = pd.DataFrame(pd.get_dummies(df[columnName], prefix=columnName))
                        df2 = df2.join(ohe_df)
                        df2.drop(columnName, axis=1, inplace=True)
                    elif input_ohe == 'Satırı sil':
                        df2 = df2[pd.to_numeric(df2[columnName], errors='coerce').notnull()]
                #Numerik verinin olması durumu
                elif df2[columnName].isnull().sum() > 0:
                    col_info = f'{columnName} '"Kolonunda  için yapilmak isteneni seçiniz"
                    col_info2 = "%s Kolonunda %s adet kayıp veri için yapılmasını istediğinizi seçiniz." \
                                % (columnName, df2[columnName].isnull().sum())
                    input = st.radio(col_info2, yap_na, key=selected[i])
                    if input == 'Sil':
                        df2 = df2[df[columnName].notna()]
                    elif input == 'Mean':
                        df2[columnName] = df2[columnName].fillna(df2[columnName].mean())
                    elif input == 'Most Frequent':
                        df2[columnName] = df2[columnName].fillna(df2[columnName].mode()[0])
                    elif input == 'Median':
                        df2[columnName] = df2[columnName].fillna(df2[columnName].median())
            st.header("Finalde Oluşan Veri Seti")
            st.dataframe(df2)

            #Oluşan son modelin indirme butonu
            name = upDataSet.name.replace(".csv", "")
            csv = df2.to_csv(index=False)
            st.download_button(
                label="CSV olarak indir",
                data=csv,
                file_name='Preprocessed_%s.csv' % (name),
                mime='text/csv',
            )
    else:
        st.warning('Henüz bir veri seti yüklenmedi!')