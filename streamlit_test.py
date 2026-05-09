import os
import sys

import streamlit as st

st.write(os.getcwd())
st.write(sys.path)

st.write("A - start")

import app.rag

st.write("B - rag imported")