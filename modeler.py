import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
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

        miss_value = df.isnull().sum().sum()
        noNum = no_numeric_col(df)
        noNum_count = len(noNum)
        Message(miss_value, noNum)

        #Modellemek için kayıp veri ve numerik veri kontrolu
        if (miss_value <= 0) & (noNum_count == 0):
            st.success("Veri seti Modellemeye uygundur.")
            """visualize_dataset = st.button("Veri setini Görselleşir.")
            if visualize_dataset:
                pairplot(df)"""

            #Sınıflandırma kolonunun seçilmesi ve dinamik olarak görüntülenmesi
            with st.sidebar:
                columns = list(df.columns)
                sec = st.selectbox("Sınıflandırma kolonunu seçiniz:", (columns))
                index = (columns.index(sec))
                X = pd.DataFrame(df.drop(df.columns[index], axis=1))
                Y = pd.DataFrame(df.iloc[:, index])
            st.header("Sınıflandırma Kolonu Seçildikten Sonraki Veri Seti")
            col1, col2 = st.columns((4, 1))
            with col1:
                st.write(X)
            with col2:
                st.write(Y)

            # Normalizayon aşamasının yapılıp yapılmayacağı sorulur
            with st.sidebar:
                norm = ('Evet', 'Hayır')
                normCheck = st.radio("Özellik ölçeklendirme aktif edilsin mi?", norm, index=1)

            #Normalizasyon yapılması istenirse dinamik olarak ekranda normalize edilmiş veriler görünecektir.
            if normCheck == 'Evet':
                st.header("Normalizasyon Sonrası Veri Seti")
                columnsName = list(X.columns)
                scaler = MaxAbsScaler()
                dfnorm = pd.DataFrame(scaler.fit_transform(X))
                dfnorm.columns = [columnsName[i] for i in range(len(dfnorm.columns))]
                dfYaz((dfnorm))
            #Train & Test Split oranının belirlenmesi
            with st.sidebar:
                dataRatio = st.slider("Test Oranını Belirleyiniz:", 1, 99, 20)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=dataRatio / 100, random_state=1)
                st.info(f'%{dataRatio}')

            #Sınıflandırmanın yapılması için kullanıcıya SelectBox üzerinden seçenekler sunulur
                selectClassifier = st.selectbox("İstediğiniz Sınıflandırma Algoritmasını Seçiniz",
                                                ('---', 'Random Forest', 'KNN', 'SVM', 'Naive Bayes',
                                                 'Logistic Regression', 'Decision Tree'))

            #Her bir sınıflandırma seçeğinin 4 hiper parametresi kullanıcıdan alınıyor(Default değerleri sklearn ile aynıdır)
                if selectClassifier == "Random Forest":
                    hyp_header(selectClassifier)
                    n_estimators = st.number_input("n_estimators", min_value=1, value=100, step=1)
                    criterion = st.selectbox("criterion", ("gini", "entropy", "log_loss"))
                    min_samples_split_RFC = st.number_input("min_samples_leaf", min_value=0, value=2)
                    min_samples_leaf_RFC = st.number_input("min_samples_leaf", min_value=0, value=1)

                if selectClassifier == "KNN":
                    hyp_header(selectClassifier)
                    n_neighbors = st.number_input("n_neighbors", min_value=1, value=5, step=1)
                    weights = st.selectbox("weights", ("uniform", "distance"))
                    algorithm = st.selectbox("algorithm", ("auto", "ball_tree", "kd_tree","brute"))
                    leaf_size = st.number_input("min_samples_leaf", min_value=1, value=30, step=1)

                if selectClassifier == "SVM":
                    hyp_header(selectClassifier)
                    C = st.number_input("C", min_value=0.01, value=1.0)
                    kernel = st.selectbox("kernel", ("linear", "poly", "rbf", "sigmoid", "precomputed"))
                    degree = st.number_input("degree", min_value=0, value=3, step=1)
                    gamma = st.selectbox("gamma", ("scale", "auto"))

                if selectClassifier == "Logistic Regression":
                    hyp_header(selectClassifier)
                    penalty = st.selectbox("penalty", ("l1", "l2", "elasticnet", "none"), index=1)
                    C = st.number_input("C", min_value=0.01, value=1.0)
                    intercept_scaling = st.number_input("intercept_scaling", min_value=0.01, value=1.0)
                    max_iter = st.number_input("max_iter", min_value=1, value=100, step=1)

                if selectClassifier == "Decision Tree":
                    hyp_header(selectClassifier)
                    criterion = st.selectbox("criterion", ("gini", "entropy", "log_loss"))
                    splitter = st.selectbox("splitter", ("best", "random"))
                    min_samples_split_DT = st.number_input("min_samples_split", min_value=0, value=2)
                    min_samples_leaf_DT = st.number_input("min_samples_leaf", min_value=0, value=1)

                if selectClassifier == "Naive Bayes":
                    NBMetods = ("Gaussian Naive Bayes", "Multinomial Naive Bayes",
                                "Complement Naive Bayes", "Bernoulli Naive Bayes")
                    NBCheck = st.radio("Bir Naive Bayes metodu seçiniz:", NBMetods)

            #Hiper parametrelerden sonraki işlem Onay butanuna basılmasıdır.
                onay = st.button("Onayla")
            #Onaylanan seçenekler seçimlere göre classifier.py'a giderek modelleme yapılmaktadır.
            if onay:
                try:
                    if selectClassifier == 'Random Forest':
                        RandomForest(X_train, Y_train, n_estimators, criterion, min_samples_split_RFC, min_samples_leaf_RFC)

                    elif selectClassifier == 'KNN':
                        KNN(X_train, Y_train, n_neighbors, weights, algorithm, leaf_size)

                    elif selectClassifier == 'SVM':
                        SVM(X_train, Y_train, C, kernel, degree, gamma)

                    elif selectClassifier == 'Naive Bayes':
                        NaiveBayes(X_train, Y_train, NBCheck)

                    elif selectClassifier == 'Logistic Regression':
                        LogisticReg(X_train, Y_train, penalty, C, intercept_scaling, max_iter)

                    elif selectClassifier == 'Decision Tree':
                        DecisionTree(X_train, Y_train, criterion, splitter, min_samples_split_DT, min_samples_leaf_DT)
                except Exception as e:
                    st.error(e)

            #Modelleme oluşturulduğunda oluşan modelin bilgilerinin kullanıcıya gösterilmesi
            try:
                clf_model = trained_model.get_model()
                st.header("Eğitilen Modelin Performansı")
                st.success(getClassifierName(clf_model))
                performanceMetrics(Y_test, X_test, clf_model)
                userInput(columns)
                btnPred = st.button("Tahminle")
                if btnPred:
                    userPrediction(clf_model, inputArray)
                downloadBtn = st.button("Modeli İndir")
                if downloadBtn:
                    download_model(clf_model)
            except:
                st.warning("Model oluşturulmadı")
        else:
            st.error("Bu Veri seti ile modelleme işlemi yapamazsınız. Lütfen Preprocessing işlemini gerçekleştirin!")
    else:
        st.warning('Henüz bir veri seti yüklenmedi!')