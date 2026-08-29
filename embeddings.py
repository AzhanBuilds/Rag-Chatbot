import os
import faiss
import numpy as np
import pickle

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag import load_documents, split_text


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


client = genai.Client(api_key=api_key)


def create_embeddings(texts):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )

    return np.array(
        [embedding.values for embedding in result.embeddings],
        dtype="float32"
    )


# Load documents
documents = load_documents()


# Split documents into chunks
all_chunks = []

for document in documents:
    chunks = split_text(document)
    all_chunks.extend(chunks)


print("Number of chunks:", len(all_chunks))


# Create Gemini embeddings
embeddings = create_embeddings(all_chunks)

print("Embedding dimensions:", embeddings.shape[1])


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print("FAISS index created!")
print("Number of vectors:", index.ntotal)


# Create vectorstore folder
os.makedirs("vectorstore", exist_ok=True)


# Save FAISS index
faiss.write_index(
    index,
    "vectorstore/index.faiss"
)


# Save chunks
with open("vectorstore/chunks.pkl", "wb") as file:
    pickle.dump(all_chunks, file)


print("FAISS index saved!")
print("Chunks saved!")