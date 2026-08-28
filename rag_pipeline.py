import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from gemini_service import ask_gemini


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS index
index = faiss.read_index("vectorstore/index.faiss")


# Load document chunks
with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


def answer_question(question):

    # 1. Convert question into embedding
    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")


    # 2. Search FAISS
    distances, indices = index.search(
        question_embedding,
        3
    )


    # 3. Get relevant chunks
    relevant_chunks = []

    for i in indices[0]:

        if i < len(chunks):
            relevant_chunks.append(chunks[i])


    # 4. Combine chunks
    context = "\n\n".join(relevant_chunks)


    # 5. Create RAG prompt
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


    # 6. Send context + question to Gemini
    answer = ask_gemini(prompt)


    return answer


# Test the complete RAG pipeline
if __name__ == "__main__":

    question = "what is Artifitial Intelligence?"

    answer = answer_question(question)

    print("\nQuestion:")
    print(question)

    print("\nRAG Answer:")
    print(answer)