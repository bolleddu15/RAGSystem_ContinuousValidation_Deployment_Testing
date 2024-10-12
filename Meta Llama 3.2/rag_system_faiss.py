from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import os


api_token = "hf_utiukEBMXxJD0sSFtexbGshnUbnjwfUzen"  
login(api_token)


model_name = "distilgpt2"  
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")


llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded successfully.")

print("Setting up FAISS index...")
index_dir = "./faiss_index"
if os.path.exists(index_dir):
    vector_store = FAISS.load_local(index_dir, embedding_model.embed_query)
else:
    raise ValueError(f"FAISS index directory {index_dir} does not exist. Run the ingestion script first to create the index.")

print("FAISS index loaded successfully.")

retriever = vector_store.as_retriever()


qa_chain = load_qa_chain(llm, chain_type="stuff")


def query_rag_system(user_query):
    print(f"Querying RAG system with query: {user_query}")
    docs = retriever.get_relevant_documents(user_query)
    result = qa_chain.run(input_documents=docs, question=user_query)
    return result


query = "What are the main principles of optical lithography?"
result = query_rag_system(query)

print("\nAnswer:", result)
