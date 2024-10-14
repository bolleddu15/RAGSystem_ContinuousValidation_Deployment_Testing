import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import nltk

nltk.download('punkt')

retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_chunk_documents():
    files = ['webpage_text_1.txt', 'pdf_text.txt', 'webpage_text_3.txt']
    documents = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            sentences = nltk.sent_tokenize(text)
            chunks = [" ".join(sentences[i:i + 15]) for i in range(0, len(sentences), 15)]  
            documents.extend(chunks)
    return documents

def encode_documents(documents):
    return retrieval_model.encode(documents, convert_to_tensor=True)

def retrieve_document(query, document_embeddings, documents, top_k=5):
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)
    results = []
    for idx in top_results.indices:
        results.append({
            'document': documents[idx],
            'score': similarities[idx].item()
        })
    return results

def answer_query_with_generative_model(query, context):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    
    inputs = tokenizer(f"Answer this question: {query} based on the context: {context}", return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=3, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    documents = load_and_chunk_documents()
    document_embeddings = encode_documents(documents)

    queries = [
        "What should be the ideal post-bake temperature to ensure thermal stability of resist?",
        "What are the main Photoresist stripping techniques?",
        "What are the advantages of using positive resist over negative resist?"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        retrieved_docs = retrieve_document(query, document_embeddings, documents, top_k=5)
        combined_context = " ".join([result['document'] for result in retrieved_docs])
        answer = answer_query_with_generative_model(query, combined_context)
        print(f"Answer: {answer}")
