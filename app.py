
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

st.title('Decision_surfaces for KNN')

da=st.selectbox('Choose dataset',['classification', 'blobs', 'moons', 'circles'])
k=st.slider("Choose no of neighbors", 1, 40, 1)
w=st.selectbox('Choose weight',['uniform','distance'])

if da=='make_classification':
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_repeated=0,random_state=20)
    knn=KNeighborsClassifier(n_neighbors=k,weights=w)









