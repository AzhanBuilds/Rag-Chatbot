import os


def load_documents():

    documents = []

    folder = "documents"

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        if filename.endswith(".txt"):

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append(text)

    return documents


def split_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks

documents = load_documents()

all_chunks = []

for document in documents:

    chunks = split_text(document)

    all_chunks.extend(chunks)


print("Number of documents:", len(documents))
print("Number of chunks:", len(all_chunks))

for i, chunk in enumerate(all_chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)