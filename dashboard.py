from multiOptions import Multioption
import streamlit as st
from options import inference
from options import eda
from options import training


st.title("Email Spam Detection Dashboard")
st.sidebar.title("Data Themes")

app = Multioption()

app.add_option('EDA', eda.app)
app.add_option('Training', training.app)
app.add_option('Prediction', inference.app)

app.run()
