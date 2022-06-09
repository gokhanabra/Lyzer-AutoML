import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from _Layout import layout
from myScripts import *
from classifier import *



def app():
    st.title("Modelling Islemleri")
    layout()

    # Dataset upload
    upDataSet = st.sidebar.file_uploader(label='Veri setini buraya yukleyiniz', type='csv')

    if upDataSet != None:
        df = pd.read_csv(upDataSet)
        dfYaz(df)

        st.write("DataFramede bulunan toplam NaN veri : ", f'{df.isnull().sum().sum()}')
        if df.isnull().sum().sum() > 0:
            st.error("Bu Veri seti ile modelleme islemi yapamazssiniz. Lutfen Preprocessing islemini gerceklestirin!")
        else:
            st.success("Veri seti Modellemeye uygundur.")

            # st.bar_chart(df)
            # fig, ax = plt.subplots()
            # ax.hist(df, bins=10)
            # st.pyplot(fig)

            with st.sidebar:

                st.write("Veri setini kolon index numaralarina gore ayiriniz")
                X1 = st.number_input('X1', min_value=0.0, max_value=float(len(df.columns)), value=0.0, step=1.0)
                X2 = st.number_input('X2', min_value=0.0, max_value=float(len(df.columns) - 2), value=1.0, step=1.0)
                Y = st.number_input('Y', min_value=-1.0, max_value=float(len(df.columns)), value=-1.0, step=1.0)
                # buton onayi iste sonra ayir.
                X = pd.DataFrame(df.iloc[:, int(X1): int(X2) + 1])
                Y = pd.DataFrame(df.iloc[:, int(Y)])

                # Normalizayon asamasinin istenmesi
                st.warning("Ozellik olceklendirme aktif edilsin mi?")
                norm = ('Evet', 'Hayir')
                normCheck = st.radio("", norm, index=1)
                isNormCheck = False
                if normCheck == 'Evet':
                    isNormCheck = True

            if isNormCheck == True:
                st.write("Normalize edilmis veri seti")
                columnsName = list(X.columns)
                transformer = Normalizer()
                dfnorm = pd.DataFrame(transformer.transform(X))
                dfnorm.columns = [columnsName[i] for i in range(len(dfnorm.columns))]
                dfYaz((dfnorm))

            with st.sidebar:

                # y = pd.DataFrame(Y)
                # dfYaz(y)
                dataRatio = st.slider("Test oranini Belirleyiniz", 1, 99, 20)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=dataRatio / 100, random_state=1)
                st.write(f'{float(dataRatio) / 100}')

                selectClassifier = st.selectbox("Istediniz Siniflandirma Algoritmasini Seciniz",
                                                ('Random Forest', 'KNN', 'SVM', 'Naive Bayes',
                                                 'Logistic Regression', 'Decision Tree'))

                if selectClassifier == "Naive Bayes":
                    NBMetods = ("Gaussian Naive Bayes", "Multinomial Naive Bayes",
                                "Complement Naive Bayes", "Bernoulli Naive Bayes")
                    NBCheck = st.radio("Bir Naive Bayes motodu seciniz", NBMetods)

                onay = st.button("Onayla")
            if onay:
                if selectClassifier == 'Random Forest':
                    with st.sidebar:
                        RandomForest(X_train, X_test, Y_train, Y_test)
                        download_model(RFC)
                    performanceMetrics(Y_test, X_test, selectClassifier)

                elif selectClassifier == 'KNN':
                    with st.sidebar:
                        KNN(X_train, X_test, Y_train, Y_test)
                        download_model(Knn)
                    performanceMetrics(Y_test, X_test, selectClassifier)

                elif selectClassifier == 'SVM':
                    with st.sidebar:
                        SVM(X_train, X_test, Y_train, Y_test)
                        download_model(Svm)
                    performanceMetrics(Y_test, X_test, selectClassifier)

                elif selectClassifier == 'Naive Bayes':
                    with st.sidebar:
                        NaiveBayes(X_train, X_test, Y_train, Y_test, NBCheck)
                        download_modelNB(NBCheck)
                    performanceMetricsNB(Y_test, X_test, NBCheck)

                elif selectClassifier == 'Logistic Regression':
                    with st.sidebar:
                        LogisticRegression(X_train, X_test, Y_train, Y_test)
                        download_model(RFC)
                    performanceMetrics(Y_test, X_test, selectClassifier)

                elif selectClassifier == 'Decision Tree':
                    with st.sidebar:
                        DecisionTree(X_train, X_test, Y_train, Y_test)
                        download_model(RFC)
                    performanceMetrics(Y_test, X_test, selectClassifier)

                """
                     wantPred = st.button("Model uzerinden tahminleme yapak icin tiklayin.")
                if wantPred:
                    inputData = list()
                    for i in range(len(X)):
                        input = st.number_input(f'{X[i]}', min_value=0.0, value=0.0, step=1.0)
                        inputData.append(input)
                    st.write(inputData)
                else:
                    pass
                """
        #Modelin yuklenmesi testi edilmesi
        modelinput = st.file_uploader("Yukle", type="pkl")
        try:
            model = pd.read_pickle(modelinput)
            results = model.score(X_test, Y_test)
            st.write(results)
        except Exception as e:
            st.error(e)

        #pred = model.predict(X_test)

    else:
        st.warning('Henuz bir veri seti yuklenmedi!')
