import fitz  
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document  
import faiss
import os


index_dir = "./faiss_index"
os.makedirs(index_dir, exist_ok=True)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


embedding_dim = 384  
index = faiss.IndexFlatL2(embedding_dim)
docstore = InMemoryDocstore({})  
index_to_docstore_id = {}


vector_store = FAISS(embedding_model.embed_query, index, docstore, index_to_docstore_id)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def add_text_to_faiss(text, source):
    
    doc = [Document(page_content=text, metadata={"source": source})]
    vector_store.add_documents(doc)


pdf_text = extract_text_from_pdf("downloaded_document.pdf")
add_text_to_faiss(pdf_text, "downloaded_document.pdf")


from langchain_community.document_loaders import WebBaseLoader
def load_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

website_docs = load_website("https://www.lithoguru.com/scientist/lithobasics.html")
vector_store.add_documents(website_docs)

vector_store.save_local(index_dir)
