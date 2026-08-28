from sentence_transformers import SentenceTransformer
from rag import load_documents, split_text
import faiss
import numpy as np
import pickle
import os


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load documents
documents = load_documents()


# Split documents into chunks
all_chunks = []

for document in documents:
    chunks = split_text(document)
    all_chunks.extend(chunks)


# Create embeddings
embeddings = model.encode(all_chunks)


# Convert embeddings to float32
embeddings = np.array(embeddings).astype("float32")


# Get the number of dimensions
dimension = embeddings.shape[1]

print("Number of chunks:", len(all_chunks))
print("Embedding dimensions:", dimension)


# Create FAISS index
index = faiss.IndexFlatL2(dimension)


# Add embeddings to FAISS
index.add(embeddings)


print("FAISS index created!")
print("Number of vectors:", index.ntotal)


# Create vectorstore folder if it doesn't exist
os.makedirs("vectorstore", exist_ok=True)


# Save FAISS index
faiss.write_index(
    index,
    "vectorstore/index.faiss"
)


# Save the text chunks
with open("vectorstore/chunks.pkl", "wb") as file:
    pickle.dump(all_chunks, file)


print("FAISS index saved!")
print("Chunks saved!")