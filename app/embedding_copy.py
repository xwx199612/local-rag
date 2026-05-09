import streamlit as st
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL
import numpy as np

_model = None
# cache the model for reuse, avoid reloading every time
@st.cache_resource
def get_model():
    global _model
    if _model is None:
        st.write("Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed(texts):
    model = get_model()
    return np.array(model.encode(texts), dtype="float32")