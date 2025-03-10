
from ast import Div
from ctypes import alignment
from re import M
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
from mlxtend.plotting import plot_learning_curves
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.markdown('<h1 style="text-align: center;">KNN Decision and learning curve</h1>', unsafe_allow_html=True)

## adding logo

st.image('logo.jpg', use_container_width=True)

da=st.selectbox('Choose dataset',['classification', 'blobs', 'moons', 'circles'])
k=st.slider("Choose no of neighbors", 1, 40, 1)
w=st.selectbox('Choose weight',['uniform','distance'])
al=st.selectbox('Choose algorithm',['ball_tree', 'kd_tree', 'brute'])

if st.button('Submit'):
    if da=='classification':
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_repeated=0,random_state=20)
        knn=KNeighborsClassifier(n_neighbors=k,weights=w,algorithm=al)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
        knn.fit(X_train,y_train)

        plt.figure(figsize=(10,10))
        st.header('Decision surface',divider='rainbow')
        plot_decision_regions(X, y,knn)
        st.pyplot(plt)

        plt.figure(figsize=(10,10))
        st.header('Learning curve',divider='rainbow')
        plot_learning_curves(X_train,y_train,X_test,y_test,knn,scoring='accuracy')
        st.pyplot(plt)
    
    elif da=='blobs':
        X, y = make_blobs(n_samples=1000, n_features=2, random_state=20)
        knn=KNeighborsClassifier(n_neighbors=k,weights=w,algorithm=al)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
        knn.fit(X_train,y_train)

        plt.figure(figsize=(10,10))
        st.header('Decision surface',divider='rainbow')
        plot_decision_regions(X, y,knn)
        st.pyplot(plt)

        plt.figure(figsize=(10,10))
        st.header('Learning curve',divider='rainbow')
        plot_learning_curves(X_train,y_train,X_test,y_test,knn,scoring='accuracy')
        st.pyplot(plt)

    elif da=='moons':
        X,y=make_moons(n_samples=1000,noise=0.1,random_state=20)
        knn=KNeighborsClassifier(n_neighbors=k,weights=w,algorithm=al)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        knn.fit(X_train,y_train)
   
        plt.figure(figsize=(10,10))
        st.header('Decision surface',divider='rainbow')
        plot_decision_regions(X, y,knn)
        st.pyplot(plt)

        plt.figure(figsize=(10,10))
        st.header('Learning curve',divider='rainbow')
        plot_learning_curves(X_train,y_train,X_test,y_test,knn,scoring='accuracy')
        st.pyplot(plt)

    elif da=='circles':
        X,y = make_circles(n_samples=1000,noise=0.1,random_state=20)
        knn=KNeighborsClassifier(n_neighbors=k,weights=w,algorithm=al)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
        knn.fit(X_train,y_train)
        
        plt.figure(figsize=(10,10))
        st.header('Decision surface',divider='rainbow')
        plot_decision_regions(X, y,knn)
        st.pyplot(plt)

        plt.figure(figsize=(10,10))
        st.header('Learning curve',divider='rainbow')
        plot_learning_curves(X_train,y_train,X_test,y_test,knn,scoring='accuracy')
        st.pyplot(plt)
    
    elif da=='blobs':
        X,y=make_blobs(n_samples=1000, n_features=2, random_state=20)
        knn=KNeighborsClassifier(n_neighbors=k,weights=w,algorithm=al)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
        knn.fit(X_train,y_train)
     
        plt.figure(figsize=(10,10))
        st.header('Decision surface',divider='rainbow')
        plot_decision_regions(X, y,knn)
        st.pyplot(plt)

        plt.figure(figsize=(10,10))
        st.header('Learning curve',divider='rainbow')
        plot_learning_curves(X_train,y_train,X_test,y_test,knn,scoring='accuracy')
        st.pyplot(plt)








