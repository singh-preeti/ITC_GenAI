import streamlit as st
import PyPDF2
import os
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    return model.encode(chunks)

def answer_query(query, chunks, chunk_embeddings):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    best_idx = np.argmax(similarities)
    return chunks[best_idx]

st.title("PDF Chatbot with Streamlit")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF and preparing chatbot..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = get_chunks(text)
        chunk_embeddings = embed_chunks(chunks)
    st.success("PDF processed! Ask your questions below.")
    query = st.text_input("Ask a question about the PDF:")
    if query:
        answer = answer_query(query, chunks, chunk_embeddings)
        st.write("**Answer:**", answer)
else:
    st.info("Please upload a PDF to get started.")