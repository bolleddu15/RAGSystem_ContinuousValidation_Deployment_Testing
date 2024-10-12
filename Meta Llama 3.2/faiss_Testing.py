import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np  


faiss_index = faiss.read_index("document_faiss.index")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)


embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar_document(query):
    print(f"Encoding query: {query}")
    query_embedding = embedder.encode(query)
    
    print("Searching FAISS index...")
    D, I = faiss_index.search(np.array([query_embedding]), k=1)  
    
    if I[0][0] != -1:
        doc_info = documents[I[0][0]]
        print(f"Document retrieved: {doc_info['text'][:200]}...")  
        return doc_info["text"], doc_info["source"]
    else:
        print("No document found in FAISS index.")
        return None, None


query = "Tell me about Southern California Edison GRC Proceedings."
document_text, source = retrieve_similar_document(query)

if document_text:
    print(f"Retrieved Document: {document_text[:500]}...\nSource: {source}")
else:
    print("No relevant document found.")
