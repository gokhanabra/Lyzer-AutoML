import pickle
import base64
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sn

#Final modelinin global olarak tutulması ve işlenmesi için Class yapısında tutulmuştur.
class model:
    def __init__(self, model = None):
        self._model = model

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

trained_model = model()
inputArray = list()

#Sınıflandırma Algritmasının adını döndürür.
def getClassifierName(estimator):
    clfname = estimator.__class__.__name__
    return clfname

#Random Forest Classification hiper parametreler ile fit edilmesi ve modelin tutulması
def RandomForest(X_train, Y_train, n_estimators, criterion, min_samples_split, min_samples_leaf):
    RFC = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf)
    RFC_model = RFC.fit(X_train, Y_train.values.ravel())
    trained_model.set_model(RFC_model)

def KNN(X_train, Y_train, n_neighbors, weights, algorithm, leaf_size):
    Knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size)
    Knn_model = Knn.fit(X_train, Y_train.values.ravel())
    trained_model.set_model(Knn_model)

def SVM(X_train, Y_train, C, kernel, degree, gamma):
    Svm = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    Svm_model = Svm.fit(X_train, Y_train.values.ravel())
    trained_model.set_model(Svm_model)

#Naive Bayes sınıflandırması adı atındaki diğer bayes teoremlerinin işleme alınması ve modelin fit edilmesi
def NaiveBayes(X_train, Y_train, metod):
    try:
        if metod == "Gaussian Naive Bayes":
            Gaussian = GaussianNB()
            Gaussian_model = Gaussian.fit(X_train, Y_train.values.ravel())
            trained_model.set_model(Gaussian_model)
        elif metod == "Multinomial Naive Bayes":
            Multinomial = MultinomialNB()
            Multinomial_model = Multinomial.fit(X_train, Y_train.values.ravel())
            trained_model.set_model(Multinomial_model)

        elif metod == "Complement Naive Bayes":
            Complement = ComplementNB()
            Complement_model = Complement.fit(X_train, Y_train.values.ravel())
            trained_model.set_model(Complement_model)

        elif metod == "Bernoulli Naive Bayes":
            Bernoulli = BernoulliNB()
            Bernoulli_model = Bernoulli.fit(X_train, Y_train.values.ravel())
            trained_model.set_model(Bernoulli_model)
    except Exception as e:
        err_mess ="Bu model gelistirilmek icin uygun degildir.\n" + f'\n{e}'
        st.error(err_mess)

def LogisticReg(X_train, Y_train, penalty, C, intercept_scaling, max_iter):
    LogisticReg = LogisticRegression(penalty=penalty, C=C, intercept_scaling=intercept_scaling, max_iter=max_iter)
    LogisticReg_model = LogisticReg.fit(X_train, Y_train.values)
    trained_model.set_model(LogisticReg_model)

def DecisionTree(X_train, Y_train, criterion, splitter, min_samples_split, min_samples_leaf):
    DecisionTreeClf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf)
    DecisionTreeClf_model = DecisionTreeClf.fit(X_train, Y_train.values.ravel())
    trained_model.set_model(DecisionTreeClf_model)

#İstenildiğnde kullanıcılar model oluşturulduktan sonra modeli pickle formatında indirebilmektedir.
def download_model(model):
    name = f'{getClassifierName(model)}'
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{name}Model.pkl">Eğitilen Modeli .pkl Olarak İndir.</a>'
    st.markdown(href, unsafe_allow_html=True)

#Model yüklenmesi fakat henüz bir işlevi yok
def upload_model(model):
    load_data = st.file_uploader("Modeli yukleme", type='pkl')

#Modelin perofrmans metriklerini hesaplamaya yarar
def performanceMetrics(Y_test, X_test,model):

    y_pred = model.predict(X_test)
    y_true = Y_test
    getMatrix(y_true, y_pred)
    getMetrics(y_true, y_pred)

#Confusion matrix verilerini seaborn ile görselleştirilmiş olarak verir
def getMatrix (y_true, y_pred):
    conf = pd.DataFrame(confusion_matrix(y_true, y_pred))
    fig = plt.figure()
    sn.heatmap(conf, annot=True, cmap='Blues', fmt='g')
    plt.title("Confusion Matrix")
    st.pyplot(fig)

#ACC REC PREC ve F1 skorlararını seaborn ile ekrana getirir.
def getMetrics(y_true, y_pred):
    class_count = len(y_true.value_counts())

    if class_count == 2:
        avg_metod = 'binary'
    else:
        avg_metod = 'macro'

    metrics = {'Accuracy': [accuracy_score(y_true, y_pred)*100],
               'Recall': [recall_score(y_true, y_pred, average=avg_metod)*100],
               'Precision': [precision_score(y_true, y_pred, average=avg_metod)*100],
               'F1': [f1_score(y_true, y_pred, average=avg_metod)*100]
               }
    metrics = pd.DataFrame(metrics)

    fig2 = plt.figure()
    sn.barplot(data=metrics)
    plt.title("Performance Metrics")
    plt.ylabel("%")
    st.pyplot(fig2)

#Pairplot olarak görselleştirme.
def pairplot(df):
    fig3 = plt.figure()
    sn.set_theme(style="darkgrid")
    sn.pairplot(df)
    st.pyplot(fig3)

#Kulanıcıdan modelleme sonrası modelin üzerinden tahminleme yapabilir. Bunun için kullanıcın girdiği verileri tutar
def userInput(columns):
    st.header("Model Üzerinden Tahminleme Yap")
    inputArray.clear()
    for i in range(len(columns) - 1):
        inputArray.append(0)
        inputArray[i] = st.number_input(columns[i])

#Kullanıcıdan gelen bilgileri model üzerinden predict eder ve tahmin sonuvunu gösterir
def userPrediction(model, inputArray):
    try:
        pred = model.predict([inputArray])
        pred_msg = "Modelin Tahmin Ettiği Etiket Değeri: %s" %(pred[0])
        st.info(pred_msg)
    except Exception as e:
        st.warning(e)
        st.error("Model Bulunamadi")