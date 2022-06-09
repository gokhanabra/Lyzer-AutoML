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
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import streamlit as st


RFC = RandomForestClassifier()
Knn = KNeighborsClassifier()
Svm = svm.SVC()
Gaussian = GaussianNB()
Multinomial = MultinomialNB()
Complement = ComplementNB()
Bernoulli = BernoulliNB()
Categorical = CategoricalNB()
LogisticReg = LogisticRegression()
DecisionTreeClf = DecisionTreeClassifier()

def RandomForest(X_train, X_test, Y_train, Y_test):
    RFC.fit(X_train, Y_train.values.ravel(), )
    st.write("Basari orani: %", str(accuracy_score(Y_test, RFC.predict(X_test))*100))
    """performanceMetrics(Y_test, X_test, classifier)"""

def KNN(X_train, X_test, Y_train, Y_test):
    Knn.fit(X_train, Y_train.values.ravel())
    st.write("Basari orani: %", str(accuracy_score(Y_test, Knn.predict(X_test)) * 100))

def SVM(X_train, X_test, Y_train, Y_test):
    Svm.fit(X_train, Y_train.values.ravel())
    st.write("Basari orani: %", str(accuracy_score(Y_test, Svm.predict(X_test)) * 100))

def NaiveBayes(X_train, X_test, Y_train, Y_test, metod):
    if metod == "Gaussian Naive Bayes":
        Gaussian.fit(X_train, Y_train.values.ravel())
        st.write("Basari orani: %", str(accuracy_score(Y_test, Gaussian.predict(X_test)) * 100))

    elif metod == "Multinomial Naive Bayes":
        Multinomial.fit(X_train, Y_train.values.ravel())
        st.write("Basari orani: %", str(accuracy_score(Y_test, Multinomial.predict(X_test)) * 100))

    elif metod == "Complement Naive Bayes":
        Complement.fit(X_train, Y_train.values.ravel())
        st.write("Basari orani: %", str(accuracy_score(Y_test, Complement.predict(X_test)) * 100))

    elif metod == "Bernoulli Naive Bayes":
        Bernoulli.fit(X_train, Y_train.values.ravel())
        st.write("Basari orani: %", str(accuracy_score(Y_test, Bernoulli.predict(X_test)) * 100))

def LogisticRegression(X_train, X_test, Y_train, Y_test):
    LogisticReg.fit(X_train, Y_train.values.ravel())
    st.write("Basari orani: %", str(accuracy_score(Y_test, LogisticReg.predict(X_test)) * 100))

def DecisionTree(X_train, X_test, Y_train, Y_test):
    DecisionTreeClf.fit(X_train, Y_train)
    st.write("Basari orani: %", str(accuracy_score(Y_test, DecisionTreeClf.predict(X_test)) * 100))


def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="trainedModel.pkl">Eğitilen Modeli .pkl Olarak İndir.</a>'
    st.markdown(href, unsafe_allow_html=True)

def upload_model(model):
    load_data = st.file_uploader("Modeli yukleme", type='pkl')

    pass
def download_modelNB(metod):
    if metod == "Gaussian Naive Bayes":
        model = Gaussian
    if metod == "Multinomial Naive Bayes":
        model = Multinomial
    if metod == "Complement Naive Bayes":
        model = Complement
    if metod == "Bernoulli Naive Bayes":
       model = Bernoulli

    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="trainedModel.pkl">Eğitilen Modeli .pkl Olarak İndir.</a>'
    st.markdown(href, unsafe_allow_html=True)

def newUserInput(X):
    inputData = list()
    for i in range(len(X)):
        input = st.number_input(f'{X[i]}', min_value=0.0, value=0.0, step=1.0)
        inputData.append(input)
    st.write(inputData)


def Prediction(new_input):
    pass

def newUserOutput():
    pass




def performanceMetrics(Y_test, X_test, classifier):
    if classifier == "Random Forest":  # Random Forest
        y_pred = RFC.predict(X_test)

    elif classifier == "KNN":  # K-nearest Neighbors
        y_pred = Knn.predict(X_test)

    elif classifier == "SVM":  # Supper Vector Machine
        y_pred = Svm.predict(X_test)

    elif classifier == "Logistic Regression":  # Logistic Regression
        y_pred = LogisticReg.predict(X_test)

    elif classifier == "Decision Tree":  # Decision Tree
        y_pred = DecisionTreeClf.predict(X_test)

    y_true = Y_test
    getMetrics(y_true, y_pred)

    # write komutu yerine dic yapisi kullanilabilir.

def performanceMetricsNB(Y_test, X_test, metod):
    if metod == "Gaussian Naive Bayes":
        y_pred = Gaussian.predict(X_test)

    if metod == "Multinomial Naive Bayes":
        y_pred = Multinomial.predict(X_test)

    if metod == "Complement Naive Bayes":
        y_pred = Complement.predict(X_test)

    if metod == "Bernoulli Naive Bayes":
        y_pred = Bernoulli.predict(X_test)

    y_true = Y_test
    getMetrics(y_true, y_pred)


def getMetrics (y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = conf[0, 0], conf[1, 1], conf[1, 0], conf[0, 1]
    conf = pd.DataFrame(conf)
    conf.columns = ['Positive', 'Negative']
    conf.index = ['Positive', 'Negative']
    st.dataframe(conf)
    acc = (TP + TN) / (TP + TN + FP + FN)
    st.write("Accuracy Score = ", acc)
    recall = TP / (TP + FN)
    st.write("Recall Score = ", recall)
    precision = TP / (TP + FP)
    st.write("Precision Score = ", precision)
    F1 = 2 * (precision * recall) / (precision + recall)
    st.write("F1 Score = ", F1)