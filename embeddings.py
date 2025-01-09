from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from preprocess import *

def generate_embeddings(chunks):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = embedding_model.encode(chunks)
    return chunk_embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index