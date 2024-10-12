import os
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk

# Download NLTK punkt tokenizer
nltk.download('punkt')

def load_and_chunk_documents():
    files = ['webpage_text_1.txt', 'pdf_text.txt', 'webpage_text_3.txt']
    documents = []
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            sentences = nltk.sent_tokenize(text)
            documents.extend(sentences)
    
    return documents

def encode_documents(documents, model):
    return model.encode(documents, convert_to_tensor=True)

def retrieve_document(query, document_embeddings, documents, model, top_k=2):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)

    results = []
    for idx in top_results.indices:
        results.append({
            'document': documents[idx],
            'score': similarities[idx].item()
        })
    return results

def answer_query_with_qa(query, document):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    qa_input = {
        'question': query,
        'context': document
    }
    answer = qa_pipeline(qa_input)
    return answer

# Streamlit app definition
st.title('Document Retrieval and Question Answering System')

# Load documents and model
st.write("Loading documents and model...")
documents = load_and_chunk_documents()
model = SentenceTransformer('all-mpnet-base-v2')
document_embeddings = encode_documents(documents, model)
st.write("Documents and model loaded successfully!")

# Input query
query = st.text_input("Enter your query:", "What is lithography?")

if st.button("Search"):
    if query:
        # Retrieve relevant documents
        retrieved_docs = retrieve_document(query, document_embeddings, documents, model, top_k=2)

        st.write("Retrieved Documents (before QA extraction):")
        for result in retrieved_docs:
            st.write(f"Score: {result['score']:.4f}")
            st.write(f"Document: {result['document'][:300]}...")

            # Answer the query using QA pipeline
            qa_answer = answer_query_with_qa(query, result['document'])
            st.write(f"Extracted Answer: {qa_answer['answer']}")
            st.write(f"Confidence: {qa_answer['score']:.4f}")
    else:
        st.write("Please enter a query to proceed.")
