import faiss
import pickle
import numpy as np

from google import genai
from google.genai import types
from gemini_service import ask_gemini
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# Create Gemini client
client = genai.Client(api_key=api_key)


# Load FAISS index
index = faiss.read_index("vectorstore/index.faiss")


# Load document chunks
with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


def create_query_embedding(question):

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )

    return np.array(
        result.embeddings[0].values,
        dtype="float32"
    ).reshape(1, -1)


def answer_question(question):

    # 1. Create Gemini embedding for the question
    question_embedding = create_query_embedding(question)


    # 2. Search FAISS
    distances, indices = index.search(
        question_embedding,
        3
    )


    # 3. Get relevant chunks
    relevant_chunks = []

    for i in indices[0]:

        if i >= 0 and i < len(chunks):
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


    # 6. Send to Gemini
    answer = ask_gemini(prompt)

    return answer


if __name__ == "__main__":

    question = "What is Artificial Intelligence?"

    answer = answer_question(question)

    print("\nQuestion:")
    print(question)

    print("\nRAG Answer:")
    print(answer)