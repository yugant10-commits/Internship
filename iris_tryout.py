import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""

# Simple Iris Classifier Application

""")
iris = datasets.load_iris()
X =iris.data
Y = iris.target

model = RandomForestClassifier()
model.fit(X,Y)


st.sidebar.header("Input flower sizes")

def user_input():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

user_df = user_input()

st.subheader("User input parameters")
st.write(user_df)

prediction = model.predict(user_df)
prediction_probability = model.predict_proba(user_df)

st.subheader('Class Labels')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])