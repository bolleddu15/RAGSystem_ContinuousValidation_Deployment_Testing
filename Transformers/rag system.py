import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk


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

if __name__ == "__main__":
    documents = load_and_chunk_documents()
    model = SentenceTransformer('all-mpnet-base-v2')
    document_embeddings = encode_documents(documents, model)
    query = "what should be the ideal post bake temperature to insure thermal stability of resist?"

    retrieved_docs = retrieve_document(query, document_embeddings, documents, model, top_k=2)

    print("Retrieved Documents (before QA extraction):")
    for result in retrieved_docs[:2]:
        print(f"Score: {result['score']:.4f}, Document: {result['document'][:300]}...")

        qa_answer = answer_query_with_qa(query, result['document'])
        print(f"Extracted Answer: {qa_answer['answer']}")
