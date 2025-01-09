import streamlit as st
from preprocess import extract_text_from_pdf, split_text_into_chunks
from embeddings import generate_embeddings, create_faiss_index
from gemini_integration import setup_gemini, generate_answer
import numpy as np

# Load PDF and preprocess
pdf_path = "KUAIC_Constituion(for RAG).pdf"
text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(text)

# Generate embeddings and create FAISS index
chunk_embeddings = generate_embeddings(chunks)
index = create_faiss_index(chunk_embeddings)

# Set up Gemini
gemini_api_key = "Your-API-KEY" 
model = setup_gemini(gemini_api_key)

# Streamlit app
st.title("KUAIC Constitution Chatbot")

# Input for user query
query = st.text_input("Ask a question about the club constitution:")

if query:
    # Retrieve relevant chunks
    query_embedding = generate_embeddings([query])
    distances, indices = index.search(np.array(query_embedding), k=2)  # Retrieve top 2 chunks
    relevant_chunks = [chunks[i] for i in indices[0]]

    # Generate answer using Gemini
    context = "\n".join(relevant_chunks)
    answer = generate_answer(query, context, model)

    # Display answer
    st.write("**Answer:**")
    st.write(answer)