from rag import load_documents

documents = load_documents()

print("Number of documents:", len(documents))

print("\nDocument content:")
print(documents[0])