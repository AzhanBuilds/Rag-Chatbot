import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from gemini_service import ask_gemini


# Load FAISS index
index = faiss.read_index("vectorstore/index.faiss")


# Load document chunks
with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


# Model will be loaded only when needed
model = None


def get_model():
    global model

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


def answer_question(question):

    # 1. Load embedding model when needed
    embedding_model = get_model()

    # 2. Convert question into embedding
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")


    # 3. Search FAISS
    distances, indices = index.search(
        question_embedding,
        3
    )


    # 4. Get relevant chunks
    relevant_chunks = []

    for i in indices[0]:

        if i < len(chunks):
            relevant_chunks.append(chunks[i])


    # 5. Combine chunks
    context = "\n\n".join(relevant_chunks)


    # 6. Create RAG prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
provide an answer but tell it is not from the context.

Context:
{context}

Question:
{question}

Answer:
"""


    # 7. Send context + question to Gemini
    answer = ask_gemini(prompt)

    return answer


if __name__ == "__main__":

    question = "what is Artificial Intelligence?"

    answer = answer_question(question)

    print("\nQuestion:")
    print(question)

    print("\nRAG Answer:")
    print(answer)