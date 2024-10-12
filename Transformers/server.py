from flask import Flask, request, jsonify, render_template
import os
from document_store_and_retrieval import load_and_chunk_documents, encode_documents, retrieve_document, answer_query_with_qa
from sentence_transformers import SentenceTransformer

flask_app = Flask(__name__)


print("Loading and encoding documents...")
documents = load_and_chunk_documents()
model = SentenceTransformer('all-mpnet-base-v2')
document_embeddings = encode_documents(documents, model)
print("Documents loaded and encoded!")


@flask_app.route('/')
def home():
    return render_template('frontend.html')  # Serve the frontend HTML file

@flask_app.route('/ask', methods=['POST'])
def ask_query():
    data = request.get_json()
    query = data.get('query')

    # Retrieve relevant documents and extract the answer
    top_docs = retrieve_document(query, document_embeddings, documents, model, top_k=2)
    for doc in top_docs:
        answer = answer_query_with_qa(query, doc['document'])

        # Send the answer back to the frontend
        return jsonify({
            'answer': answer['answer'],
            'confidence': answer['score']
        })

if __name__ == '__main__':
    # Set the template folder to the current directory if necessary
    flask_app.template_folder = os.getcwd()
    flask_app.run(debug=True)
