from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS index
index = faiss.read_index("vectorstore/index.faiss")


# Load saved text chunks
with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


# Ask a question
question = "What is Artificial Intelligence?"


# Convert question into an embedding
question_embedding = model.encode([question])

question_embedding = np.array(question_embedding).astype("float32")


# Search FAISS
distances, indices = index.search(
    question_embedding,
    1
)


# Display result
print("\nQuestion:")
print(question)

print("\nMost relevant document:")
print(chunks[indices[0][0]])

print("\nDistance:")
print(distances[0][0])